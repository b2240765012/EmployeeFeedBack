"""
Metron - Veri Modelleri
------------------------
AI'ın üreteceği çıktının kesin JSON şemasını garanti altına almak için
Pydantic modelleri burada tanımlanır. LangChain'in `with_structured_output`
metodu bu modeli kullanarak LLM'i JSON Schema / function-calling üzerinden
zorlar; serbest metin üretme ihtimali ortadan kalkar.
"""

from typing import List, Literal
from pydantic import BaseModel, Field


class MetronAnalysisOutput(BaseModel):
    """Metron AI'ın üst yönetime sunduğu günlük değerlendirme çıktısı."""

    employee_id: str = Field(
        ..., description="Çalışanın benzersiz kimlik numarası veya kullanıcı adı."
    )
    daily_performance_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Günlük performans skoru, 1 (çok düşük) ile 10 (mükemmel) arası.",
    )
    performance_status: Literal[
        "Yüksek", "Dengeli", "Dışsal Engelli", "Verimsizlik Riski"
    ] = Field(..., description="Günün genel durum sınıflandırması.")
    summary_for_management: str = Field(
        ...,
        description=(
            "Üst yönetimin 10 saniyede okuyabileceği; profesyonel, yapıcı, "
            "objektif ve kanıta dayalı özet metin. Suçlayıcı dil içermez."
        ),
    )
    blockers_detected: List[str] = Field(
        default_factory=list,
        description="Çalışandan kaynaklanmayan, tespit edilen dışsal engellerin listesi.",
    )
    burnout_stress_level: Literal["Düşük", "Orta", "Yüksek"] = Field(
        ..., description="Metnin dil tonundan sezilen stres / tükenmişlik riski."
    )
    trend_analysis: str = Field(
        ...,
        description=(
            "Bugünkü performansın, çalışanın son 30 günlük geçmişine göre "
            "yorumlanmış hali (örn. 'Geçmiş performansına paralel', "
            "'Beklenmedik düşüş', 'Kronik verimsizlik eğilimiyle örtüşüyor')."
        ),
    )


class EmployeeHistory(BaseModel):
    """Girdi olarak kullanılan 30 günlük geçmiş performans özeti."""

    avg_score_last_30_days: float = Field(..., ge=0, le=10)
    dominant_stress_trend: Literal["Düşük", "Orta", "Yüksek"] = "Düşük"
    past_notes: str = ""
