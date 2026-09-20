def get_app() -> WebApp:
    app = _STATE.get("app")
    if app is not None:
        return app 
    from realaiagent.config import Config
    from realaiagent.engine import Agent
    import os
    cfg = Config.from_env()  # auto /tmp/realai-data when VERCEL=1
    agent = Agent(cfg)
    try: agent.keys.ensure_owner_key()
    except: pass
    master = None
    if agent.telegram.enabled:
        from realaiagent.telegram import TelegramMaster
        master = TelegramMaster(agent.telegram, agent, agent.users)
        if not os.getenv("VERCEL"):
            master.start()  # polling only locally
    from realaiagent.web import WebApp
    app = WebApp(agent)  # <- no master arg in this repo
    app._tg_master = master
    _STATE["app"] = app
    return app
