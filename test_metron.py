"""
Metron - Örnek Test Senaryosu
-------------------------------
Bu script, MetronEngine'i 3 farklı gerçekçi senaryo ile çalıştırır:

  1) Ahmet   -> Dışsal engel (altyapı arızası + uzayan toplantı), dürüst rapor.
  2) Zeynep  -> Yüksek performans, geçmişle uyumlu, tutarlı dijital iz.
  3) Can     -> Rapor ile dijital iz arasında belirgin tutarsızlık
                (manipülasyon/tutarsızlık sinyali) + geçmişte de zayıf trend.

Çalıştırmadan önce ortam değişkenini ayarlayın:
    export OPENAI_API_KEY="sk-..."

Çalıştırma:
    python test_metron.py
"""

import json
import os

from metron_engine import MetronEngine


def print_result(title: str, result) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY ortam değişkeni bulunamadı. "
            "Lütfen 'export OPENAI_API_KEY=sk-...' ile ayarlayın."
        )

    engine = MetronEngine(model_name="gpt-4o", temperature=0.2)

    # -----------------------------------------------------------------
    # SENARYO 1: Dışsal Engel (Altyapı Sorunu + Uzayan Toplantı)
    # -----------------------------------------------------------------
    ahmet_report = (
        "Bugün sabah CI/CD pipeline'ı 3 saat boyunca kırık kaldı, DevOps "
        "ekibiyle uğraştık ama deploy edemedik. Öğleden sonra da müşteri "
        "tarafından planlanan 1 saatlik toplantı 2.5 saate uzadı. Kalan "
        "zamanda PR-482 üzerinde küçük düzeltmeler yaptım ve review'lara "
        "cevap verdim."
    )
    ahmet_github = [
        {"commit_message": "fix: PR-482 review yorumlarına küçük düzeltme", "files_changed": 2},
    ]
    ahmet_jira = [
        {"ticket": "INFRA-901", "status": "Blocked", "description": "CI/CD pipeline arızası - DevOps'a escalate edildi"},
        {"ticket": "PR-482", "status": "In Review", "description": "Review yorumları işlendi"},
    ]
    ahmet_history = {
        "avg_score_last_30_days": 8.1,
        "dominant_stress_trend": "Düşük",
        "past_notes": "Genellikle tutarlı ve yüksek performans gösteriyor.",
    }

    result_1 = engine.analyze(
        employee_id="AHM-1042",
        employee_report=ahmet_report,
        github_logs=ahmet_github,
        jira_logs=ahmet_jira,
        employee_history=ahmet_history,
    )
    print_result("SENARYO 1: Dışsal Engelli Gün (Ahmet)", result_1)

    # -----------------------------------------------------------------
    # SENARYO 2: Yüksek ve Tutarlı Performans
    # -----------------------------------------------------------------
    zeynep_report = (
        "Bugün auth modülündeki refactor'u tamamladım, 4 unit test ekledim "
        "ve PR-510'u merge'e hazır hale getirdim. Ayrıca sprint planlama "
        "toplantısına katıldım, 2 yeni task aldım."
    )
    zeynep_github = [
        {"commit_message": "refactor: auth modülü temizliği", "files_changed": 6},
        {"commit_message": "test: auth modülü için 4 yeni unit test", "files_changed": 3},
        {"commit_message": "chore: PR-510 merge hazırlığı", "files_changed": 1},
    ]
    zeynep_jira = [
        {"ticket": "AUTH-77", "status": "Done", "description": "Refactor tamamlandı, testler eklendi"},
        {"ticket": "AUTH-81", "status": "To Do", "description": "Sprint planlamada alındı"},
    ]
    zeynep_history = {
        "avg_score_last_30_days": 9.0,
        "dominant_stress_trend": "Düşük",
        "past_notes": "Ekip içinde en istikrarlı performansçılardan biri.",
    }

    result_2 = engine.analyze(
        employee_id="ZYN-2077",
        employee_report=zeynep_report,
        github_logs=zeynep_github,
        jira_logs=zeynep_jira,
        employee_history=zeynep_history,
    )
    print_result("SENARYO 2: Yüksek Performans (Zeynep)", result_2)

    # -----------------------------------------------------------------
    # SENARYO 3: Rapor - Dijital İz Tutarsızlığı + Kronik Zayıf Trend
    # -----------------------------------------------------------------
    can_report = (
        "Bugün gerçekten çok yoğundum, sabahtan akşama kadar kritik ödeme "
        "modülündeki bugı çözmeye çalıştım, gece geç saatlere kadar uğraştım "
        "ve sonunda çözdüm. Yarın merge edeceğim."
    )
    can_github: list = []  # Hiç commit yok
    can_jira = [
        {"ticket": "PAY-233", "status": "In Progress", "description": "Durum güncellenmedi, yorum yok"},
    ]
    can_history = {
        "avg_score_last_30_days": 4.8,
        "dominant_stress_trend": "Orta",
        "past_notes": "Son 3 ayda benzer 'rapor-log uyumsuzluğu' iki kez daha not edilmişti.",
    }

    result_3 = engine.analyze(
        employee_id="CAN-3391",
        employee_report=can_report,
        github_logs=can_github,
        jira_logs=can_jira,
        employee_history=can_history,
    )
    print_result("SENARYO 3: Tutarsızlık Sinyali (Can)", result_3)


if __name__ == "__main__":
    main()
