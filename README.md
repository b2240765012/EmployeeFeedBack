# Metron — Akıllı Mesai Değerlendirme Sistemi

Remote/hibrit ekiplerin günlük eforunu, blokajlarını ve tükenmişlik riskini
analiz eden, üst yönetime yapılandırılmış JSON çıktısı üreten backend sistemi.

## Dosya Yapısı
```
metron_ai/
├── models.py          # Pydantic çıktı şeması (MetronAnalysisOutput)
├── metron_engine.py   # Ana motor + detaylı system prompt (MetronEngine)
├── test_metron.py     # 3 mock senaryo ile uçtan uca test
├── requirements.txt
└── README.md
```

## Kurulum
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ortam Değişkeni
```bash
export OPENAI_API_KEY="sk-..."
```

## Çalıştırma
```bash
python test_metron.py
```

## Programatik Kullanım
```python
from metron_engine import MetronEngine

engine = MetronEngine(model_name="gpt-4o", temperature=0.2)

result = engine.analyze(
    employee_id="EMP-001",
    employee_report="Bugün X modülünü geliştirdim...",
    github_logs=[{"commit_message": "feat: X modülü", "files_changed": 5}],
    jira_logs=[{"ticket": "PROJ-12", "status": "Done", "description": "..."}],
    employee_history={
        "avg_score_last_30_days": 7.5,
        "dominant_stress_trend": "Düşük",
        "past_notes": "İstikrarlı performans.",
    },
)

print(result.model_dump_json(indent=2))
```

## Tasarım Notları
- **Structured Output garantisi:** `ChatOpenAI.with_structured_output(MetronAnalysisOutput)`
  kullanılır; bu, OpenAI'nin native function-calling/JSON-schema mekanizmasını
  devreye sokar ve modelin şema dışı çıktı üretmesini engeller.
- **Adalet ilkesi:** Sistem promptu, belirsiz durumları çalışan lehine
  yorumlamayı, dışsal engelleri izole etmeyi ve manipülasyon şüphesini
  suçlayıcı değil kurumsal/nötr bir dille ("süreç ve dijital ayak izi
  tutarsızlığı") raporlamayı zorunlu kılar.
- **Nihai karar insanda kalır:** Çıktı her zaman bir *ön analiz* niteliğindedir;
  "Verimsizlik Riski" gibi etiketler bir suçlama değil, yöneticinin doğrulaması
  gereken bir sinyaldir.
- **Model seçimi:** Varsayılan `gpt-4o`; maliyet/hız için `gpt-4o-mini` de
  kullanılabilir (yapılandırılmış çıktıyı destekleyen tüm modeller uyumludur).

## Genişletme Fikirleri
- `employee_history`'yi bir veritabanından (ör. PostgreSQL) otomatik çekmek.
- Haftalık/aylık toplu rapor için `analyze()` çıktısını bir zaman serisi
  tablosuna yazıp trend grafikleri üretmek.
- `burnout_stress_level` "Yüksek" çıkan durumlarda otomatik olarak İK'ya
  bilgilendirme e-postası tetiklemek (insan onayı ile).
