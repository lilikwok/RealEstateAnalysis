/** 2. update the year_built in improvement_builtas**/
update improvement_builtas
set year_built = adjusted_year_built
where adjusted_year_built >  year_built;

go 

/* 3. only remain the latest built records*/
create view v_improvement_builtas_update
as
select *
from
(select ROW_NUMBER() over (partition by parcel_number, building_id order by year_built desc) as lastyear, *
from improvement_builtas) m
where lastyear = 1 ;

 go

/* 4. delete the adjusted_year_built, lastyear, built_as_number from improvement_builtas_update*/


/* 6. join improvement_builtas_update and improvement table */

create view v_full_improvement
as
select i.[parcel_number]
      ,i.[building_id]
      ,[square_feet]
      ,[percent_complete]
      ,[condition]
      ,[quality]
      ,[attic_finished_square_feet]
      ,[basement_square_feet]
      ,[basement_finished_square_feet]
      ,[porch_square_feet]
      ,[attached_garage_square_feet]
      ,[detached_garage_square_feet]
      ,[fireplaces]
      ,[stories]
      ,[bedrooms]
      ,[bathrooms]
      ,[year_built]
from improvement i left join v_improvement_builtas_update u
on  i.building_id = u.building_id and i.parcel_number = u.parcel_number;

go

create view v_appraisal_account 
as
select Parcel_Number as aa_Parcel_Number, Appraisal_Account_Type, Land_Economic_Area, Land_Net_Acres, Appraisal_Date, Utility_Sewer, Utility_Water, Utility_Electric, View_Quality, Waterfront_Type
from dbo.appraisal_account;

go

create view v_tax_account
as 
select Parcel_Number as ta_Parcel_Number, Total_Market_Value_Current_Year, Total_Market_Value_Prior_Year, Range, Township
from dbo.tax_account;

go

create view v_improvement
as 
select Parcel_Number as i_Parcel_Number, Square_Feet, Percent_Complete, Condition, Quality, Attic_Finished_Square_Feet, Attached_Garage_Square_Feet, Basement_Finished_Square_Feet, Basement_Square_Feet, Detached_Garage_Square_Feet, Fireplaces, Porch_Square_Feet
from dbo.improvement;

go

create view v_improvement_builtas
as 
select Parcel_Number as ib_Parcel_Number, Stories, Bedrooms, Bathrooms, Year_Built, Adjusted_Year_Built
from dbo.improvement_builtas;

go

create view v_sale
as 
select ETN,Parcel_Number as s_Parcel_Number, Sale_Date, Sale_Price
from dbo.sale;

go

create view v_sale_2018
as 
select ETN,Parcel_Number as s_Parcel_Number, Sale_Date, Sale_Price
from dbo.sale
where Sale_Date between '2018-01-01' and '2018-12-31';

go

create view v_resi_appraisal_account 
as
select Parcel_Number as aa_Parcel_Number, Appraisal_Account_Type, Land_Economic_Area, Land_Net_Acres, Appraisal_Date, Utility_Sewer, Utility_Water, Utility_Electric, View_Quality, Waterfront_Type
from dbo.appraisal_account
where dbo.appraisal_account.Appraisal_Account_Type = 'Residential';

go

create view v_master_table_mid_product
as
select vs.Sale_Date, sale_price, vs.s_Parcel_Number as Parcel_Number, vraa.Land_Economic_Area, vraa.Land_Net_Acres, vraa.Appraisal_Date,vraa.Utility_Sewer, 
vraa.Utility_Water, vraa.Utility_Electric, vraa.View_Quality, vraa.Waterfront_Type, vfi.Attached_Garage_Square_Feet, vfi.Attic_Finished_Square_Feet, vfi.Basement_Square_Feet, 
vfi.Bathrooms, vfi.Bedrooms, vfi.Condition, vfi.Detached_Garage_Square_Feet, vfi.Fireplaces, vfi.Porch_Square_Feet,vfi.Quality, vfi.Square_Feet, vfi.Stories, vfi.Year_Built,
cd.Crime_Num, ISNULL(si.withInSewerImprovement, 'N') AS withInSewerImprovement, ISNULL(nf.near_firestation, 'N') AS near_firestation,
ISNULL(nh.near_healthcare, 'N') AS near_healthcare, ISNULL(nl.near_libraries, 'N') AS near_libraries, ISNULL(np.near_policestation, 'N') AS near_policestation,
ISNULL(ns.near_schools, 'N') as near_schools, ISNULL(nw.near_waterplants, 'N') as near_waterplants
from dbo.v_sale_2018 as vs 
	left join dbo.v_resi_appraisal_account as vraa on vs.s_Parcel_Number = vraa.aa_Parcel_Number
	left join dbo.v_full_improvement as vfi on vs.s_Parcel_Number = vfi.Parcel_Number
	left join dbo.v_tax_account as vta on vs.s_Parcel_Number = vta.ta_Parcel_Number
	left join dbo.crime_data as cd on (vta.Range = cd.Range and vta.Township = cd.Township)
	left join dbo.IN_SEWER_IMPROVEMENT_DISTRICT as si on vs.s_Parcel_Number = si.taxparceln
	left join dbo.NEAR_FIRESTATION as nf on vs.s_Parcel_Number = nf.taxparceln
	left join dbo.NEAR_HEALTHCARE as nh on vs.s_Parcel_Number = nh.taxparceln
	left join dbo.NEAR_LIBRARIES as nl on vs.s_Parcel_Number = nl.taxparceln
	left join dbo.NEAR_POLICESTATION as np on vs.s_Parcel_Number = np.taxparceln
	left join dbo.NEAR_SCHOOLS as ns on vs.s_Parcel_Number = ns.taxparceln
	left join dbo.NEAR_WATERPLANTS as nw on vs.s_Parcel_Number = nw.taxparceln;

go

create view v_master_table_final
as 
select * from (
SELECT *, ROW_NUMBER() OVER (
 PARTITION BY Parcel_Number
 ORDER BY Sale_Date) AS row_num
FROM 
v_master_table_mid_product) as t
where t.row_num = 1
