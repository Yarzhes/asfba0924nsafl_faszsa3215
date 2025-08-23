import json
from ultra_signals.core.config import load_settings, ConfigError

def redact_secrets(settings_dict):
    """Recursively redacts sensitive keys in a nested dictionary."""
    sensitive_keys = ["api_key", "api_secret", "bot_token"]
    for key, value in settings_dict.items():
        if key in sensitive_keys:
            settings_dict[key] = "[REDACTED]"
        elif isinstance(value, dict):
            redact_secrets(value)
    return settings_dict

def main():
    """
    Loads settings, redacts secrets, and prints the result.
    """
    try:
        settings = load_settings()
        settings_dict = settings.model_dump()
        
        redacted_settings = redact_secrets(settings_dict)
        
        print("Settings loaded and validated successfully.")
        print(json.dumps(redacted_settings, indent=2))
        
    except ConfigError as e:
        print(f"Configuration validation failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()