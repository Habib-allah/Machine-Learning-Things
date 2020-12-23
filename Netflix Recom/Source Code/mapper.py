from operator import itemgetter
class Mapper():
  def __init__(self,df):
    self.profile_dict,self.asset_dict=self.map(df)

  def map(self,df):
    '''
    maps user ids and asset ids to indices starting from 0.
    
    df_train: a part of df_interest

    return: two dictionaries
    '''
    profiles_set=set(df['id_profile'])
    profile_dict = dict(zip(profiles_set, range(len(profiles_set))))

    asset_set=set(df['id_asset'])
    asset_dict = dict(zip(asset_set, range(len(asset_set))))

    return profile_dict,asset_dict

  def get_many(self,keys,dictionary='profile'):

    if dictionary=='profile':
      d=self.profile_dict
    elif dictionary=='asset':
      d=self.asset_dict
    else:
      return None

    if (isinstance(keys,float)) or (isinstance(keys,int)): return [d[keys]]
    return list(itemgetter(*keys)(d))