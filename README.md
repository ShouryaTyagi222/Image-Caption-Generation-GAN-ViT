
## Model training
To train the model, execute the following command:
```sh
$ python main.py config.json
```

## Model test
To test the model, execute the following command:
```sh
$ python test.py config.json
```
This script will create two output files :
* `output_argmax` : contains the  generated captions obtained by selecting arg max words.
* `output_beam` : contains the  generated captions obtained by using beam search.

To get Top 5 and Flop 5 for one of these files, execute the following command :
```sh
$ python evaluator.py cocodataset/captions/beam.en output_argmax cocodataset/links/beam.txt cocodatset/images
```
That will generate 10 PNG files containing these results.
