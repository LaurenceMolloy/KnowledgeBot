"""Application configuration settings with single source of truth"""
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, ClassVar, Dict, Any
from .Resolver import ConfigResolver

# Single source of truth for all defaults
DEFAULTS = {
    # LLM Settings
    "ENABLE_LLM": False,
    "LLM_PROVIDER": "ollama",
    "LLM_MODEL": "phi3:mini", 
    "LLM_SERVER": "localhost",
    "LLM_PORT": 11434,
    
    # Slack Settings
    "SLACK_THREAD_MAX_AGE_DAYS": 7,
    "SLACK_EDIT_CHANNEL": "test_edit",
    "SLACK_KNOWLEDGE_CHANNELS": "test_knowledge",
    "SLACK_CHANNEL_TYPES": "public_channel",
    "SLACK_BOT_EMOJI": "mortar_board",
    
    # Vector Database Settings
    "VECTOR_DB_PROVIDER": "weaviate",
    "VECTOR_DB_SERVER": "localhost",
    "VECTOR_DB_PORT": 8080,
    "VECTOR_DB_CHUNK_LENGTH": 100,
    "VECTOR_DB_CHUNK_OVERLAP": 20,
    "VECTOR_DB_CHUNK_EMBEDDING": None,

    # File System Settings
    "EXPORT_FOLDER": "./data",
    "KNOWLEDGE_FOLDER": "./data/knowledge",
    "VECTOR_DB_FOLDER": "./data/vectordb",
    "STATE_FILE": "./data/state.json",
}

@dataclass
class BotConfig:
    """KnowledgeBot configuration with single source of truth"""
    DEFAULTS: ClassVar[Dict] = DEFAULTS
    
    # Required field (no default)
    slack_token: str
    
    # LLM Settings
    enable_llm: bool = field(default_factory=lambda: DEFAULTS["ENABLE_LLM"])
    llm_provider: str = field(default_factory=lambda: DEFAULTS["LLM_PROVIDER"])
    llm_model: str = field(default_factory=lambda: DEFAULTS["LLM_MODEL"])
    llm_server: str = field(default_factory=lambda: DEFAULTS["LLM_SERVER"])
    llm_port: int = field(default_factory=lambda: DEFAULTS["LLM_PORT"])
    
    # File System Settings
    export_folder: Path = field(default_factory=lambda: Path(DEFAULTS["EXPORT_FOLDER"]))
    state_file: Path = field(default_factory=lambda: Path(DEFAULTS["STATE_FILE"]))
    
    # Slack Settings
    slack_thread_max_age_days: int = field(default_factory=lambda: DEFAULTS["SLACK_THREAD_MAX_AGE_DAYS"])
    slack_edit_channel: str = field(default_factory=lambda: DEFAULTS["SLACK_EDIT_CHANNEL"])
    slack_knowledge_channels: List[str] = field(
        default_factory=lambda: DEFAULTS["SLACK_KNOWLEDGE_CHANNELS"].split(",")
    )
    slack_channel_types: str = field(default_factory=lambda: DEFAULTS["SLACK_CHANNEL_TYPES"])
    slack_bot_emoji: str = field(default_factory=lambda: DEFAULTS["SLACK_BOT_EMOJI"])
    
    # Vector Database Settings
    vector_db_provider: str = field(default_factory=lambda: DEFAULTS["VECTOR_DB_PROVIDER"])
    vector_db_server: str = field(default_factory=lambda: DEFAULTS["VECTOR_DB_SERVER"])
    vector_db_port: int = field(default_factory=lambda: DEFAULTS["VECTOR_DB_PORT"])
    vector_db_folder: Path = field(default_factory=lambda: Path(DEFAULTS["VECTOR_DB_FOLDER"]))
    vector_db_chunk_length: int = field(default_factory=lambda: DEFAULTS["VECTOR_DB_CHUNK_LENGTH"])
    vector_db_chunk_overlap: int = field(default_factory=lambda: DEFAULTS["VECTOR_DB_CHUNK_OVERLAP"])
    vector_db_chunk_embedding: str = field(default_factory=lambda: DEFAULTS["VECTOR_DB_CHUNK_EMBEDDING"])
    
    @classmethod
    def from_resolver(cls, config_resolver: ConfigResolver) -> "BotConfig":
        """Create config from secret manager using single source defaults"""
        return cls(
            # Required field
            slack_token=config_resolver["SLACK_TOKEN"],
            
            # LLM Settings
            enable_llm=config_resolver.get("ENABLE_LLM", str(cls.DEFAULTS["ENABLE_LLM"])).value.lower() in ("true", "1", "yes", "on"),
            llm_provider=config_resolver.get("LLM_PROVIDER", cls.DEFAULTS["LLM_PROVIDER"]).value,
            llm_model=config_resolver.get("LLM_MODEL", cls.DEFAULTS["LLM_MODEL"]).value,
            llm_server=config_resolver.get("LLM_SERVER", cls.DEFAULTS["LLM_SERVER"]).value,
            llm_port=int(config_resolver.get("LLM_PORT", str(cls.DEFAULTS["LLM_PORT"])).value),
            
            # File System Settings
            export_folder=Path(config_resolver.get("EXPORT_FOLDER", cls.DEFAULTS["EXPORT_FOLDER"]).value),
            state_file=Path(config_resolver.get("STATE_FILE", cls.DEFAULTS["STATE_FILE"]).value),
            
            # Slack Settings
            slack_thread_max_age_days=int(config_resolver.get("SLACK_THREAD_MAX_AGE_DAYS", str(cls.DEFAULTS["SLACK_THREAD_MAX_AGE_DAYS"])).value),
            slack_edit_channel=config_resolver.get("SLACK_EDIT_CHANNEL", cls.DEFAULTS["SLACK_EDIT_CHANNEL"]).value,
            slack_knowledge_channels=cls._parse_list(config_resolver.get("SLACK_KNOWLEDGE_CHANNELS", cls.DEFAULTS["SLACK_KNOWLEDGE_CHANNELS"]).value),
            slack_channel_types=config_resolver.get("SLACK_CHANNEL_TYPES", cls.DEFAULTS["SLACK_CHANNEL_TYPES"]).value,
            slack_bot_emoji=config_resolver.get("SLACK_BOT_EMOJI", cls.DEFAULTS["SLACK_BOT_EMOJI"]).value,
            
            # Vector Database Settings
            vector_db_provider=config_resolver.get("VECTOR_DB_PROVIDER", cls.DEFAULTS["VECTOR_DB_PROVIDER"]).value,
            vector_db_server=config_resolver.get("VECTOR_DB_SERVER", cls.DEFAULTS["VECTOR_DB_SERVER"]).value,
            vector_db_port=int(config_resolver.get("VECTOR_DB_PORT", str(cls.DEFAULTS["VECTOR_DB_PORT"])).value),
            vector_db_folder=Path(config_resolver.get("VECTOR_DB_FOLDER", cls.DEFAULTS["VECTOR_DB_FOLDER"]).value),
            vector_db_chunk_length=int(config_resolver.get("VECTOR_DB_CHUNK_LENGTH", str(cls.DEFAULTS["VECTOR_DB_CHUNK_LENGTH"])).value),
            vector_db_chunk_overlap=int(config_resolver.get("VECTOR_DB_CHUNK_OVERLAP", str(cls.DEFAULTS["VECTOR_DB_CHUNK_OVERLAP"])).value),
            vector_db_chunk_embedding=config_resolver.get("VECTOR_DB_CHUNK_EMBEDDING", str(cls.DEFAULTS["VECTOR_DB_CHUNK_EMBEDDING"])).value,
        )
    @classmethod
    def get_default(cls, key: str) -> Any:
        """Get a default value by key without instantiating the class"""
        return cls.DEFAULTS.get(key)

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Get all defaults without instantiating the class"""
        return cls.DEFAULTS.copy()

    @staticmethod
    def _parse_list(value: str) -> List[str]:
        """Parse comma-separated list, handling whitespace"""
        return [item.strip() for item in value.split(",") if item.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert BotConfig to a regular dictionary"""
        return {field.name: getattr(self, field.name) for field in fields(self)}
    