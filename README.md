# Food and Drink Description Generator in Bahasa
> This is my final project for Narutal Language Processing Course

Want to give it a try? Try it now on [![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://food-drink-description-generator-bahasa.streamlit.app/)

## Dataset
[Indonesia food delivery Gofood product list](https://www.kaggle.com/datasets/ariqsyahalam/indonesia-food-delivery-gofood-product-list) by Reyhan Ariq Syahalam on Kaggle. This dataset contains **45,422** Gofood product list, where **21,947** of them contain product description.

### Example
| merchant_name | merchant_area | category | display | product | price | discount_price | isDiscount | description |
| --- | --- | --- | --- |
| "330 Kopi, Ciledug" | jakarta | Kopi/Minuman/Roti | Non Coffee | Lychee Squash | 20000 | 0 |  |
| "330 Kopi, Ciledug" | jakarta | Kopi/Minuman/Roti | Non Coffee | Chocoreo | 25000 | 0 | Perpaduan Chocolate Dan Oreo |
| "330 Kopi, Ciledug" | jakarta | Kopi/Minuman/Roti | Snack | Pisang Lumer Coklat | 19000 | 0 | Sajian Pisang Lumer Dengan Rasa Coklat Yang Begitu Menggoda |

## Result
| Model  | Train Loss | Validation Loss |
| --- | --- | --- | --- |
| [LSTM](https://keras.io/api/layers/recurrent_layers/lstm/) | 2.2738 |  |
| Finetuned [Pretrained GPT2](https://huggingface.co/cahya/gpt2-small-indonesian-522M) | 1.8034 | 2.3559 |

## Project Structure
```bash
├── datasets
│   ├── gofood_dataset.csv
│   ├── train.csv
│   ├── validation.csv
│   ├── word_to_idx.pkl
│   └── idx_to_word.pkl
├── app.py
├── train.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Streamlit Demo
```bash
streamlit run app.py
```

## Author
[Rizky Adi](https://www.linkedin.com/in/rizky-adi-7b008920b/)


Last update: 22 November 2022
