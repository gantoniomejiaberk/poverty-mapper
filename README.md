# poverty-mapper


### About the Project
Poverty Mapper is a capstone project for the UC Berkeley Master of Data and Information Science program (http://povertymapper.com/). The project uses deep learning on satellite data to make predictions about poverty prevalence within and across five Asian countries: Bangladesh, Nepal, The Philippines, Tajikistan, and Timor Leste. Our mission is to help international development NGOs to make better decisions about how to allocate resources by filling gaps in poverty data.

The project utilizes transfer learning on pretrained convolutional neural networks (CNNs) and the PyTorch machine learning framework. Project data is stored on a restricted access AWS s3 bucket.

### Terms of Use
The code in this repository is available for public use without restrictions.

### Recommended Citation
Ayele, S. Mejia, G., Zorrilla, L. (2021). poverty-mapper.
https://github.com/gantoniomejiaberk/poverty-mapper

### Table of Contents
- **sentinel2_data**: Download global, cloud-free composite satellite imagery from the European Commission Joint Research Centre Data Catalogue: https://data.jrc.ec.europa.eu/dataset/0bd1dfab-e311-4046-8911-c54a8750df79. Divide into 224x224 tiles and check against UTM zone, country boundaries, and capture image meta data, including center and corner coordinates. 
- **dhs_data**: Process Demographic and Health Survey (DHS) data for use in satellite image labeling. DHS data available upon request from: https://dhsprogram.com/data/  
- **shape_files**: Download IPUMS country shapefiles: https://international.ipums.org/international/gis.shtml.
- **model**: Add distance-based weighted average labels to satellite images, bin labels, and prepare data splits for within and across country modeling. Create model specs and run files to tune hyperparameters and return results and model artifacts for model selection. Make within country predictions for all satellite images using best trained models. 
- **visualizations**: Process model results and create map of countries used in study, map of available DHS survey data, and poverty prediction map.
