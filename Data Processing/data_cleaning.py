import pandas as pd
df = pd.read_csv('./codes/master_model.csv')
df = df.drop(columns = ['near_healthcare','near_schools'])



hospital = pd.read_csv('hospital.csv')
hospital = hospital[['FID','TaxParcelN']]
hospital['near_hospital'] = 1
df = pd.merge(df,hospital,how='left',left_on='Parcel_Number', right_on='TaxParcelN')
df['near_hospital'] = df['near_hospital'].fillna(0)
df = df.drop(columns = ['TaxParcelN','FID'])





private_school = pd.read_csv('private_school.csv')
private_school = private_school[['objectid','taxparceln']]
private_school['near_private_school'] = 1
df = pd.merge(df,private_school,how='left',left_on='Parcel_Number', right_on='taxparceln')
df['near_private_school'] = df['near_private_school'].fillna(0)
df = df.drop(columns = ['objectid','taxparceln'])







elementary_school = pd.read_csv('elementary_school.csv')
elementary_school = elementary_school[['objectid','taxparceln']]
elementary_school['near_elementary_school'] = 1
df = pd.merge(df,elementary_school,how='left',left_on='Parcel_Number', right_on='taxparceln')
df['near_elementary_school'] = df['near_elementary_school'].fillna(0)
df = df.drop(columns = ['objectid','taxparceln'])




high_school = pd.read_csv('high_school.csv')
high_school = high_school[['objectid','taxparceln']]
high_school['near_high_school'] = 1
df = pd.merge(df,high_school,how='left',left_on='Parcel_Number', right_on='taxparceln')
df['near_high_school'] = df['near_high_school'].fillna(0)
df = df.drop(columns = ['objectid','taxparceln'])







college = pd.read_csv('college.csv')
college = college[['objectid','taxparceln']]
college['near_college'] = 1
df = pd.merge(df,college,how='left',left_on='Parcel_Number', right_on='taxparceln')
df['near_college'] = df['near_college'].fillna(0)
df = df.drop(columns = ['objectid','taxparceln'])


df.to_csv('M1_Full.csv', index=False)

df2 = df.drop(columns = ['withInSewerImprovement','near_firestation','near_libraries','near_policestation','near_waterplants','near_hospital','near_private_school','near_elementary_school','near_high_school','near_college'])

df2.to_csv('M2_Full.csv', index=False)