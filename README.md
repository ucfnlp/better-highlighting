# Better Highlighting: Creating Sub-Sentence Summary Highlights

We provide the source code for the paper **"[Better Highlighting: Creating Sub-Sentence Summary Highlights](https://arxiv.org/pdf/2010.10566.pdf)"**, accepted at EMNLP'20. If you find the code useful, please cite the following paper. 

    @inproceedings{cho-song-li-yu-foroosh-liu:2020,
     Author = {Sangwoo Cho and Kaiqiang Song and Chen Li and Dong Yu and  Hassan Foroosh and Fei Liu},
     Title = {Better Highlighting: Creating Sub-Sentence Summary Highlights},
     Booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
     Year = {2020}}

## Goal

- Our system seeks to summarize multi-document articles with sub-sentence segments
- The code consists of sub-sentence segment generation, segment importance and similarity score computation, and DPP. 

## Dependencies

The code is written in Python (v3.6) and Pytorch (v1.4). We suggest the following environment:

* [Python (v3.6)](https://www.anaconda.com/download/)
* [Pytorch (v1.4)](https://pytorch.org/get-started/locally/)
* [Pyrouge](https://pypi.org/project/pyrouge/)
* [spaCy](https://spacy.io/usage)
* [Huggingface Transformer](https://github.com/huggingface/transformers)

## Sub-sentence generation

* `seg_gen.py`: generate all segments from sentences with XLNet
 ` $ python seg_gen.py --dataset 0 --split train --data_start 0 --data_end 1` 

* `seg_filter_subsent.py`: filter out segments (generate candidate segments for a summary)
` $ python seg_filter_subsent.py --dataset 0`

* `draw_fullsent_pos.py`: draw positions of original sentences in percent
` $ python draw_fullsent_pos.py`

## BERT-sim, BERT-imp fine-tuning on CNN/DM
### Data generation

* We use the CNN/DM summary dataset, downloaded from [HERE](https://github.com/ucfnlp/summarization-dpp-capsnet) (pre-processed CNN/DM summary data file [direct link]((https://drive.google.com/file/d/1_c4AqnEct0HMg0VOWqupcO0_ijn-fJb0/view?usp=sharing))).
* This data contains a list of candidate summary sentences (the most similiar sentences to the summary) in each article.

* `gen_cnndm_pairs.py`: generate train/test data for BERT-sim (pair), BERT-imp (pair_leadn)

### BERT-sim, BERT-imp train / test

  * `run_finetune.py`: main
  * `train_finetune.py`: trainer
  * `dataset_cnndm.py`: data feeder
  * BERT-sim: ` $ python run_finetune.py --data_type pair --max_seq_len 128`
  * BERT-imp: ` $ python run_finetune.py --data_type pair_leadn --max_seq_len 512`


## BERT-sim, BERT-imp prediction on target dataset

  * `run_bert_scores.py`: BERT similarity and importance score prediction (DUC, TAC)
    ` $ python run_bert_scores.py --dataset 0 --data_type xlnet --split train --data_start 0 --data_end 1 --gpu_id 0 --batch_size_imp 5 --batch_size_sim 25`
  * `merge_bert_ext.py`: merge predicted files into one file and convert *.pkl to *.mat (BERT feature file on CNN is *.h5 and no conversion to *.mat)
    ` $ python merge_bert_ext.py --dataset 0 --data_type xlnet --split train`


## Text generation for DPP training

  * `gen_text_DPP.py`: generate texts (.txt, .words, .pos, .Y, .YY, .seg, idf, dict) for DPP training/testing from candidate segments or sentences on DUC, TAC
    ` $ python gen_text_DPP.py --dataset 0 --data_type xlnet`

## Utility files

  * `read_text_from_data.py`: text loader from DUC, TAC
  * `utils.py`: utility functions


## DPP training, testing

  * under `DPP` directory
  * `$ bash run.bash run_DPP.m`
  * `run_DPP.m`: main file to set parameters and run DPP training/testing 
  * `main_DPP.m`: read train/test text and call `DPP.m`  
  * `DPP.m`: set more specific parameters, assign features, DPP train/test

## System summary

We provide our best system summaries of DUC-04 and TAC-11 (`/summary_results`). We do not provide DUC and TAC dataset due to license. Please download [DUC 03/04](https://duc.nist.gov/) and [TAC 08/09/10/11](https://tac.nist.gov/data/index.html) dataset with your request and approval.

## License

This project is licensed under the BSD License - see the [LICENSE.md](LICENSE.md) file for details.