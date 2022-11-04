# DBA for Time Series Signal Classification

This is the official implementation of Dynamic Bilinear Low-Rank Attention (DBA) on time series signal classification.

## Requirements
```bash
pip install -r requirements.txt
mkdir experiments
mkdir dataset
```

## Dataset
Download the [dataset](http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip) and place it in the root folder as follows:

```angular2html
|-- dataset
	|-- Multivariate2018_ts
	|  |-- Multivariate_ts
	   |  |-- EthanolConcentration
       |  |-- FaceDetection
           ...
```

## Experiments
### Ethanolconcentration
```bash
bash EthanolConcentration.sh
```
### Facedetection
```bash
bash FaceDetection.sh
```
### Handwriting
```bash
bash handwriting.sh
```
### Heartbeat
```bash
bash heartbeat.sh
```
### Japanese vowels
```bash
bash japanese.sh
```
### Pems-Sf
```bash
bash pems.sh
```
### Selfregulationscp1
```bash
Selfregulationscp1.sh
```
### Selfregulationscp2
```bash
bash Selfregulationscp2.sh
```
### Spokenarabicdigits
```bash
bash SpokenArabicDigits.sh
```
### Uwavegesturelibrary
```bash
UWaveGestureLibrary.sh
```
