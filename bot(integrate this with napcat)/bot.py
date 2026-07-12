import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter
from pathlib import Path

_bot_dir = Path(__file__).resolve().parent

nonebot.init()

driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)

nonebot.load_from_toml(str(_bot_dir / "pyproject.toml"))

if __name__ == "__main__":
    nonebot.run()
