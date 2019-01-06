# bubble_noise_CNN
### ABSTRACT

The significant area are decisive location points in time-frequency spectra for a listener to correctly identify the phonetic information of the target. By adopting the method of "bubble noise" [(Mandel et al., 2016)](http://m.mr-pc.org/work/jasa16.pdf), it is proved that such area in time-frequency spectra exists for Mandarin tone identification, and the shape of the area varies for different tones. I am curious about if machines are able to learn the patterns and predict if human listeners can correctly identify the tones, too. I construct four similar convolutional neural network classifiers, each for one tone. All classifiers expect for tone 3 beat the baseline and result in satisfying performance. This paper introduces the background of current task, explains the architecture of the convolutional neural network, demonstrates the classification results, and analyze the difficulties of tone 3 prediction.

### Adapt the Scripts to Run

`path` takes the directories of all folders that contain `.mat` file feature matrics.

`result_file` takes the directories of all `.csv` files.
