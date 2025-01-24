# 123PandaRoux

Column to keep : Humidite | sismicite | date | quartier | catastrophe

What is sent by IOT : Humidite | sismicite | date (Timestamp) | quartier

IOT team give every day a CSV with all data collected during the day
    -> We have to aggregate those CSV and then get the average of the day


Mapping for "catastrophe" :
        0 - aucun | 1 - seisme | 2 - innondation | 3 - both

Power BI : To change the csv data first go in the Power Query (Transformer les données) and edit the data source (Parametre de la source de donnees)
