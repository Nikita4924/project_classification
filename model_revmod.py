import json
import hashlib

def save_model(data: dict, path: str = "model.revmod"):
    """
    Сохраняет модель в защищённом формате .revmod
    """
    raw = json.dumps(data, sort_keys=True).encode("utf-8")
    checksum = hashlib.sha256(raw).hexdigest()
    data["checksum"] = checksum

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_model(path: str = "model.revmod") -> dict:
    """
    Загружает модель и проверяет целостность
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    checksum = data.pop("checksum", None)
    raw = json.dumps(data, sort_keys=True).encode("utf-8")
    actual = hashlib.sha256(raw).hexdigest()

    if checksum != actual:
        raise ValueError("Файл модели повреждён или отредактирован вручную.")

    return data