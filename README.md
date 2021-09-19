# Gaps in Electric Vehicle Infrastructure

In the next decade, the number of electric vehicles (EVs) on our roads will likely rise substantially, and we believe the bulk of the charging of these vehicles will take place at home. But charging away from home or a workplace will also be key to support EV growth. Such on-the-go charge-ups will also need to be as easy and convenient as refueling a traditional, internal combustion engine (ICE) vehicle today. 

While EVs account for less than 2%* of new vehicle registrations in the US, and much less than 1%* of all vehicles on the road today, the underlying economics of EV ownership are improving. State and federal regulators are incentivizing adoption, signaling that EV penetration will likely increase steadily over the next decade and beyond. California—often a bellwether for other states—has moved to ban new ICE cars from 2035. We expect that, by then, EV ownership will be economically viable for most people.

So, Are we ready for EV future? Well, While EV registrations are growing at a near exponential rate in NY, the charging infrastructure does not adequately supply this demand. The ranges of EVs are increasing greatly, but the availability and speed of charging has not kept up.  

The NY state has provided a $4,000 tax credit to businesses to install new chargers. Additionally, New York City has begun installing curbside public stations to meet the demand. This motivated us to seek to identify gaps in this infrastructure and propose specific sites for development to fill these gaps. 

* IHS Reference - https://ihsmarkit.com/research-analysis/ev-registrations-exceed-2-of-overall-us-market-share-in-decemb.html

# Datasets and Data Sources

#### EV Charging Stations

**Description:** Electric Vehicle Charging stations dataset will be key driver for the project to understand the current infrastructure of EV Charging stations across US and Canada. This dataset contains alternate EV Charging stations details for the past 10 years.  
**Size:** ~16 MB. The dataset contains all US and Canada EV Charging Station records.  
**Source:** US Department of Energy (Alternate Fuels Data Center).  
**Format:** CSV  
**Access Method:** Downloaded via [AFDC Advanced filters section of the tool](https://afdc.energy.gov/fuels/electricity_locations.html#/analyze)

#### New York EV Registrations

**Description:** This dataset contains the file of vehicle, snowmobile and boat registrations in NYS. This dataset contains all types of vehicles, and we will be using only passenger vehicles including Internal Combustion Engine(ICE) vehicles, Hybrid and Electric Vehicles.  
**Size:** 1.41 GB  
**Source:** data.ny.gov   
**Format:** CSV  
**Access Method:** Data export from [data.ny.gov](https://data.ny.gov/Transportation/Vehicle-Snowmobile-and-Boat-Registrations/w4pv-hbkt)

#### New York Traffic Data

**Description:** Locations of short-duration traffic counts collected from 2015 through 2019 on roads in New York State.  
**Size:** 17 MB  
**Source:** gis.ny.gov  
**Format:** gdb  
**Access Method:** Downloaded from [gis.ny.gov](https://gis.ny.gov/gisdata/inventories/details.cfm?DSID=1280)

#### US Geographical Data

**Description:** The 2019 cartographic boundary shapefiles are simplified representations of selected geographic areas from the U.S. Census Bureau's Master Address File / Topologically Integrated Geographic Encoding and Referencing (MAF/TIGER) Database (MTDB). ZIP Code Tabulation Areas (ZCTAs) are approximate area representations of U.S. Postal Service (USPS) ZIP Code service areas.  
**Size:** 101.7 MB  
**Source:** data.gov  
**Format:** Shape Files (.shp)  
**Access Method:** Downloaded from [data.gov](https://catalog.data.gov/dataset/2019-cartographic-boundary-shapefile-2010-zip-code-tabulation-areas-for-united-states-1-500000)

#### NY EV Charging Sessions

**Description:** New York State EV Charging Session details such as charging levels, duration, connectors.  
**Size:** 30 MB  
**Source:** nyserda.ny.gov  
**Format:** Excel (xlsx) 
**Access Method:** Downloaded from [nyserda.ny.gov](https://www.nyserda.ny.gov/-/media/Files/Publications/Research/Transportation/EValuateNY-ZIP-File.zip?la=en)

# Data Manipulation and Analysis

We have performed various data manipluation steps based on the datasets. Our primary dataset (EV Charging Stations) contained all US and CA . We have used only US charging stations as our scope is to explore gaps in New York charging station infrastructure. We have multiple secondary datasets, New York vehicle registrations, US Zip and County Shape files. All of them contain geometrical data elements. The volume of the NY Vehicle registration dataset is huge (1.5GB) and we have filtered the dataset to get only Electric and Conventional passenger vehicles. So, we have used Spark to handle the volume and split-apply-combine technique is most of our data manipulation and data exploration.

Overall, the datasets that we used are holistic and good in quality. But still we had to perform basic cleanups by handling Nulls, empty strings, incomplete data values, etc., We joined the datasets by location data elements such as
Zip Code ,State, City and County Names, Latitude and Longitude coordinates.

Below segments will provide more details on the Exploratory data analysis performed on the datasets and also the algorithm used to calculate the traffic which is key for identifying the gaps is NY EV charging infrastructure.

#### NY Vehicle Registration EDA

[Exploratory Data Analysis 1 - NY State Vehicle Registrations](1_EVCharge_Registrations_abakert_aramm.ipynb)

#### NY EV Charging Stations EDA

[Exploratory Data Analysis 2 - NY EV Charging Stations](2_EVCharge_ChargingStations_abakert_aramm.ipynb)

#### NY EV Charging Sessions EDA

[Exploratory Data Analysis 3 - NY EV Charging Sessions](3_EVCharge_ChargingSessions_abakert_aramm.ipynb)

#### NY Traffic Algorithm EDA

[Exploratory Data Analysis 4 - NY Traffic](4_EVCharge_Traffic_Algorithm_abakert_aramm.ipynb)

#### NY Chosen Locations

[Algortihm - EV Traffic Calculation](5_EVCharge_Locations_abakert_aramm.ipynb)

# Conclusion

In order to evaluate these locations based on their distance to an existing charging station and their adjusted estimated traffic counts, we created an algorithm that estimates superiority. This algorithm used the data frame as well as evaluation columns as inputs, then determined to how many other locations each location was superior. A location was superior to another location if it had greater distance from a charging station and greater adjusted traffic. Using this information we created a variable, percent superior, that estimated the percent of other locations to which each location was superior. We sorted in descending order based on this metric and selected the top 10 locations. We then used reverse geocoding to find the address of each location based on their latitude and longitude. Our chosen locations are shown in the table above. Some of the locations were repetitive as they are close to each other. However, some businesses may be more willing to install charging stations than others.
