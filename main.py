from recsys.recsys import page_scenario_manager
from taipy.gui import Gui
from taipy.config import Config
import taipy as tp

Config.configure_global_app(clean_entities_enabled=True)
tp.clean_all_entities()

gui = Gui(page_scenario_manager)
gui.run()
