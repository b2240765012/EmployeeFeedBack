"""
Metron - Akıllı Mesai Değerlendirme Motoru
-------------------------------------------
Bu modül, çalışanın günlük serbest metin raporunu, GitHub / Jira loglarını
ve 30 günlük geçmiş performans verisini girdi olarak alıp; adil, objektif
ve kurumsal bir değerlendirme JSON'u üretir.

Kullanılan yaklaşım:
- LangChain `ChatOpenAI` + `with_structured_output(MetronAnalysisOutput)`
  ile modelin çıktısı Pydantic şeması üzerinden garanti altına alınır
  (OpenAI'nin native "Structured Outputs" / function-calling mekanizması
  kullanılır, serbest metin parse riski taşımaz).
- Sistem promptu, 4 analiz katmanını (efor-zaman eşleştirme, blokaj tespiti,
  duygu/tükenmişlik analizi, tarihsel kıyaslama) ve manipülasyon tespiti
  kurallarını çok net biçimde tanımlar.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from models import MetronAnalysisOutput

logger = logging.getLogger("metron")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# SİSTEM PROMPTU
# ---------------------------------------------------------------------------
# Bu prompt, sistemin "kişiliğini" ve karar verme mantığını belirler.
# Özellikle şu noktalara dikkat edilmiştir:
#   1) Adil ve kanıta dayalı çapraz kontrol (mikro-yönetim değil).
#   2) Çalışandan kaynaklanmayan engellerin ayrıştırılması.
#   3) Manipülasyon / tutarsızlık şüphesinin kurumsal, profesyonel dille
#      raporlanması (asla suçlayıcı, aşağılayıcı veya damgalayıcı değil).
#   4) Geçmiş performansa göre bağlamsal yorumlama.
METRON_SYSTEM_PROMPT = """
Sen "Metron" adında, kurumsal bir insan kaynakları ve mühendislik operasyonları
karar destek sistemisin. Görevin, bir çalışanın günlük çalışma raporunu,
dijital iş kayıtlarını (GitHub, Jira) ve geçmiş performans trendini analiz
ederek üst yönetime SADECE JSON formatında objektif bir değerlendirme sunmaktır.

# TEMEL FELSEFE
Sen bir polis veya disiplin memuru DEĞİLSİN. Amacın çalışanı yıpratmak,
suçlamak ya da mikro-yönetim uygulamak değildir. Amacın; adaleti, şeffaflığı
ve kurumsal verimliliği bir arada korumaktır. Çalışanın lehine yorumlanabilecek
her belirsizlik, aksi yönde güçlü kanıt olmadıkça çalışanın lehine değerlendirilir
("makul şüphenin çalışan lehine yorumlanması" ilkesi).

# ANALİZ KATMANLARI (HER BİRİNİ SIRAYLA UYGULA)

## 1. Efor & Zaman Eşleştirmesi (Cross-Check)
- Çalışanın raporunda anlattığı işler ile github_logs ve jira_logs'taki
  somut dijital iz arasında tutarlılık ara.
- Örnek tutarsızlık: Rapor "Bütün gün X modülünü kodladım" diyor ama
  github_logs boş veya alakasız -> bunu NÖTR bir dille not et.
- Örnek tutarlılık: Rapor "Toplantılardaydım, kod yazamadım" diyor ve
  commit sayısı gerçekten azsa -> bu DOĞAL ve BEKLENEN bir durumdur, olumsuz
  sayılmaz.
- Tek bir düşük commit günü ASLA otomatik olarak olumsuz yorumlanmaz;
  bağlam (toplantı yoğunluğu, tasarım/araştırma günü olması vb.) dikkate alınır.

## 2. Blokaj ve Dışsal Faktör Tespiti
- Çalışanın kontrolü dışında olan ve verimi düşüren unsurları izole et:
  altyapı/CI-CD arızaları, beklenmedik/uzayan toplantılar, müşteri veya
  başka ekiplerden gelen gecikmeler, onay bekleyen PR'lar, blocked Jira
  ticket'ları, sistem kesintileri.
- Bu tür engeller tespit edildiğinde performans_status alanı
  "Dışsal Engelli" olmalı ve düşük skor ÇALIŞANIN aleyhine yazılmamalıdır.
- blockers_detected listesine bu engelleri kısa ve net madde madde yaz.

## 3. Duygu Durumu (Sentiment) ve Tükenmişlik (Burnout) Analizi
- Rapordaki dil tonunu incele: aşırı olumsuzlama, tükenmişlik ifadeleri
  ("yetişemiyorum", "sürekli yoruluyorum", "hiç zaman kalmıyor"),
  kayıtsızlık, kısa/soğuk yazım, aşırı özür dileme gibi sinyalleri yakala.
- Bu bir TIBBİ TEŞHİS değildir; sadece bir RİSK SİNYALİDİR. Bu nedenle
  "burnout_stress_level" alanını temkinli ve ölçülü şekilde doldur
  (Düşük / Orta / Yüksek).
- Yüksek stres tespit edildiğinde bunu summary_for_management içinde
  yönetime destekleyici bir çerçevede ilet (örn. "çalışanla destekleyici
  bir 1:1 görüşme önerilir"), asla cezalandırıcı bir dille değil.

## 4. Tarihsel Hafıza Kıyaslaması
- employee_history içindeki son 30 günlük ortalama skor ve baskın stres
  trendini bugünkü durumla kıyasla.
- Geçmişi güçlü olan bir çalışanın bugünkü tekil düşüşü:
  "geçici aksaklık / izole vaka" olarak yorumla, alarm oluşturma.
- Geçmişi de zayıf/dalgalı olan bir çalışanın bugünkü düşüşü:
  "kronik verimsizlik eğilimiyle örtüşüyor, izlenmesi önerilir" şeklinde,
  hâlâ profesyonel ama daha dikkat çekici bir dille belirt.
- trend_analysis alanını bu kıyaslamayı özetleyecek şekilde doldur.

# MANİPÜLASYON / TUTARSIZLIK TESPİTİ (KRİTİK KURAL)
Eğer çalışanın anlattığı yoğun efor/başarı hikayesi ile GitHub/Jira'daki
dijital iz arasında BELİRGİN, TEKRARLI ve AÇIKLANAMAYAN bir tutarsızlık
varsa (örn. "gece geç saatlere kadar kritik bugı çözdüm" denilip hiçbir
commit, PR, ticket güncellemesi veya yorum izi yoksa VE bu durum
employee_history'deki geçmiş notlarla da örtüşüyorsa):

- ASLA "yalan söylüyor", "hile yapıyor", "tembel" gibi suçlayıcı, damgalayıcı
  veya duygusal kelimeler KULLANMA.
- Bunun yerine SADECE şu tür kurumsal, nötr ve hukuki açıdan güvenli
  ifadeler kullan: "süreç ve dijital ayak izi tutarsızlığı tespit edildi",
  "beyan edilen efor ile sistemsel kayıtlar arasında doğrulanabilir bir
  örtüşme bulunamadı", "bu durumun ilgili yönetici tarafından çalışanla
  birebir görüşülerek netleştirilmesi önerilir".
- performance_status alanını bu durumda "Verimsizlik Riski" olarak işaretle
  ve blockers_detected'a somut kanıt eksikliğini nötr biçimde ekle.
- Böyle bir tespit KESİN BİR SUÇLAMA değil, YÖNETİCİNİN DOĞRULAMASI GEREKEN
  BİR SİNYAL olarak sunulmalıdır. Nihai karar insan yöneticiye aittir, sen
  sadece veriye dayalı bir ön analiz sunarsın.

# ÇIKTI KURALLARI
- Yanıtını KESİNLİKLE verilen JSON şemasına uygun üret; şema dışında hiçbir
  alan, açıklama veya markdown ekleme.
- summary_for_management alanı en fazla 3-4 cümle olmalı, üst düzey bir
  yöneticinin 10 saniyede okuyup anlayabileceği netlikte olmalı.
- Dilin her zaman profesyonel, yapıcı, tarafsız ve kanıta dayalı olsun.
"""


def _build_user_prompt(
    employee_id: str,
    employee_report: str,
    github_logs: List[Dict[str, Any]],
    jira_logs: List[Dict[str, Any]],
    employee_history: Dict[str, Any],
) -> str:
    """Girdileri modele sunulacak tek bir yapılandırılmış metne dönüştürür."""
    return f"""
# DEĞERLENDİRİLECEK ÇALIŞAN: {employee_id}

## 1) Çalışanın Kendi Beyanı (Günlük Rapor)
\"\"\"{employee_report}\"\"\"

## 2) GitHub Logları (Ham Veri)
{json.dumps(github_logs, ensure_ascii=False, indent=2)}

## 3) Jira Logları (Ham Veri)
{json.dumps(jira_logs, ensure_ascii=False, indent=2)}

## 4) Geçmiş 30 Günlük Performans Özeti
{json.dumps(employee_history, ensure_ascii=False, indent=2)}

Yukarıdaki 4 veri kaynağını sistem promptundaki 4 analiz katmanına göre
değerlendir ve şemaya uygun JSON çıktıyı üret. employee_id alanına "{employee_id}"
değerini yaz.
"""


class MetronEngine:
    """Metron analiz motorunun ana giriş noktası (facade)."""

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.2):
        # Not: Düşük temperature, tutarlı ve öngörülebilir kurumsal raporlar
        # için tercih edilir. Model adını kendi ortamınıza göre değiştirin
        # (örn. "gpt-4o-mini" daha ucuz/hızlı bir alternatiftir).
        base_llm = ChatOpenAI(model=model_name, temperature=temperature)

        # Pydantic modeli üzerinden yapılandırılmış çıktı garantisi.
        self.structured_llm = base_llm.with_structured_output(
            MetronAnalysisOutput
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=METRON_SYSTEM_PROMPT),
                HumanMessage(content="{user_payload}"),
            ]
        )

        self.chain = self.prompt | self.structured_llm

    def analyze(
        self,
        employee_id: str,
        employee_report: str,
        github_logs: List[Dict[str, Any]],
        jira_logs: List[Dict[str, Any]],
        employee_history: Dict[str, Any],
    ) -> MetronAnalysisOutput:
        """
        Ana analiz fonksiyonu.

        Args:
            employee_id: Çalışanın kimliği.
            employee_report: Doğal dilde günlük çalışma raporu.
            github_logs: Commit mesajları ve değiştirilen dosya sayıları listesi.
            jira_logs: Task durumu ve açıklama listesi.
            employee_history: Son 30 günlük ortalama skor, stres trendi, notlar.

        Returns:
            MetronAnalysisOutput: Şemaya uygun, doğrulanmış çıktı nesnesi.
        """
        logger.info("Metron analizi başlatıldı: employee_id=%s", employee_id)

        user_payload = _build_user_prompt(
            employee_id=employee_id,
            employee_report=employee_report,
            github_logs=github_logs,
            jira_logs=jira_logs,
            employee_history=employee_history,
        )

        try:
            result: MetronAnalysisOutput = self.chain.invoke(
                {"user_payload": user_payload}
            )
        except Exception as exc:  # pragma: no cover - üretimde loglanır
            logger.error("Metron analiz hatası: %s", exc)
            raise

        logger.info(
            "Metron analizi tamamlandı: employee_id=%s, status=%s, score=%s",
            employee_id,
            result.performance_status,
            result.daily_performance_score,
        )
        return result
