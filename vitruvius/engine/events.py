"""Événements aléatoires : Incendie, Sécheresse, Bonne récolte, Immigration."""

from pydantic import BaseModel, model_validator


class EventEffect(BaseModel):
    """Effet d'un événement."""

    type: str
    count: int | None = None
    immune_unique: bool = False
    modifier: float | None = None
    min_amount: int | None = None
    max_amount: int | None = None


class EventPrevention(BaseModel):
    """Mécanisme de prévention d'un événement."""

    building: str
    risk_divisor: int


class EventConfig(BaseModel):
    """Configuration d'un événement."""

    display_name: str
    probability: float
    duration: int
    effect: EventEffect
    prevention: EventPrevention | None = None


class EventsConfig(BaseModel):
    """Configuration complète des événements chargée depuis events.yaml."""

    events: dict[str, EventConfig]

    @model_validator(mode="after")
    def validate_probabilities(self) -> "EventsConfig":
        total = sum(e.probability for e in self.events.values())
        if total >= 1.0:
            raise ValueError(
                f"La somme des probabilités d'événements ({total:.3f}) doit être < 1.0."
            )
        return self
