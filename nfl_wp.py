import pandas as pd
from nflwin import utilities
data = utilities.get_nfldb_play_data(season_years=[2010,2011,2012,2013,2014,2015,2016,2017],season_types=["Regular"])
a=pd.DataFrame(a)
a.to_csv("nfl_wp_data.csv")
from nflwin.model import WPModel
standard_model = WPModel.load_model()
wp = standard_model.predict_wp(data)
b=pd.DataFrame(wp)
b.to_csv("nfl_wp.csv")

