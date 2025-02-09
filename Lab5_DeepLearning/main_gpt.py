import pandas as pd 
import os
import csv
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import tensorflow as tf 


# ================      DATA COLLECTION & PREPARATION       ========================

# Example data: Replace this with your actual collected fragments
data = [
    {"fragment": ["Белеет парус одинокий В тумане моря голубом!.. Что ищет он в стране далекой? Что кинул он в краю родном", 
                  'Скажи-ка, дядя, ведь недаром Москва, спаленная пожаром, Французу отдана?',
                  'Выхожу один я на дорогу; Сквозь туман кремнистый путь блестит; Ночь тиха. Пустыня внемлет Богу, И звезда с звездою говорит.',
                  'Где гнутся над омутом лозы, Где тени сгущаются в рощах, Где тают и плачут берёзы В безмолвии пламенных рощах.'], 
    "author": "Михаил Лермонтов"},
    {"fragment": ["Люблю грозу в начале мая, Когда весенний, первый гром, Как бы резвяся и играя, Грохочет в небе голубом.", 
                  'Умом Россию не понять, Аршином общим не измерить: У ней особенная стать — В Россию можно только верить.',
                  'Нам не дано предугадать, Как наше слово отзовётся, И нам сочувствие даётся, Как нам даётся благодать.',
                  'Silentium! — Молчи, скрывайся и таи И чувства и мечты свои — Пусть в душевной глубине И восходят, и заходят Безмолвно, как звёзды в ночи…'], 
     "author": "Фёдор Тютчев"},
    {"fragment": ["Сжала руки под тёмной вуалью... 'Отчего ты сегодня бледна?' — Оттого, что я терпкой печалью Напоила его допьяна.",
                  'Не с теми я, кто бросил землю На растерзание врагам. Их грубой лести я не внемлю, Им песен я своих не дам.', 
                  'Мне ни к чему одические рати И прелесть элегических затей. По мне, в стихах всё быть должно некстати, Не так, как у людей.',
                  'Заплаканная осень, как вдова, В одеждах чёрных, всё сердца туманит... Перебирая мужнины слова, Она рыдать не перестанет.'], 
     "author": "Анна Ахматова"},
    {"fragment": ["Ты жива ещё, моя старушка? Жив и я. Привет тебе, привет! Пусть струится над твоей избушкой Тот вечерний несказанный свет.",
                  'До свиданья, друг мой, до свиданья. Милый мой, ты у меня в груди. Предназначенное расставанье Обещает встречу впереди.', 
                  'Гой ты, Русь, моя родная, Хаты — в ризах образа... Не видать конца и края — Только синь сосёт глаза.', 
                  'Не жалею, не зову, не плачу, Всё пройдёт, как с белых яблонь дым. Увяданья золотом охвачен, Я не буду больше молодым.'], 
     "author": "Сергей Есенин"},
]

data = [
    {
        "fragment": [
            "Белеет парус одинокий В тумане моря голубом!.. Что ищет он в стране далекой? Что кинул он в краю родном",
            "Скажи-ка, дядя, ведь недаром Москва, спаленная пожаром, Французу отдана?",
            "Выхожу один я на дорогу; Сквозь туман кремнистый путь блестит; Ночь тиха. Пустыня внемлет Богу, И звезда с звездою говорит.",
            "Где гнутся над омутом лозы, Где тени сгущаются в рощах, Где тают и плачут берёзы В безмолвии пламенных рощах.",
            "Пленительный, но краткий миг, Когда на троне красота...",
            "Я не унижусь пред тобою, Ни твой привет, ни твой укор",
            "Мы встретились случайно у ограды, Среди вельможных, праздных толп.",
            "Листья трепещут, листья дрожат, Звезды на небе танцуют подряд.",
            "Мой демон хмурый вновь меня терзает, В кромешной тьме теряя смысл. ",
            "Отчизна милый мой, я вновь к тебе стремлюсь!",
            "Не плачь, не плачь, мой друг, что мир жесток и тёмен",
            "Светает вновь, и лес заиграл красками утра",
            "Ты ветер слышишь в дебрях ночи? Он славу старую поёт!",
            "Мне снились берега далёкие, где солнце плавно опадает.",
            "Воспоминанья юности мне вновь ночами приходят.",
            "Не тронь природу в этот час, Она поёт и оживает.",
            "На небе облако плывёт — словно в танце вечном.",
            "Я мчусь туда, где ветер свежий, где солнца нет, но тишь и мгла.",
            "Взор мой туманит дальний свет, но сердце молчит.",
            "Грозу я видел с детства там, где небо яростно рвало ночи.",
            "Мгновенья счастья на земле так редки, но они нас держат.",
            "Кто знает, что сокрыто в тьме, где луна светит одиноко.",
            "На берегах Кавказа вновь моя душа спокойно спит.",
            "Тени лесов меня тревожат, словно память о прошлом."
        ],
        "author": "Михаил Лермонтов"
    },
    {
        "fragment": [
            "Люблю грозу в начале мая, Когда весенний, первый гром, Как бы резвяся и играя, Грохочет в небе голубом.",
            "Умом Россию не понять, Аршином общим не измерить: У ней особенная стать — В Россию можно только верить.",
            "Нам не дано предугадать, Как наше слово отзовётся, И нам сочувствие даётся, Как нам даётся благодать.",
            "Silentium! — Молчи, скрывайся и таи И чувства и мечты свои — Пусть в душевной глубине И восходят, и заходят Безмолвно, как звёзды в ночи…",
            "Природа мощная, родная, Тебе я снова гимн пою!",
            "Как капли слёз, звенят слова, Что в сердце падают, как в реку.",
            "Неведомая тишина над миром распростёрла крылья.",
            "В горах Кавказа снег сверкает, в душе моей покой.",
            "Скажи мне, свет, что будет завтра, как ночь покроет всё вокруг.",
            "Природа шепчет мне слова, что только звёзды понимают.",
            "О, как я чувствую огонь, который дарит мне рассвет!",
            "Гроза уходит, тучи тают, но след её в душе.",
            "Ты молчишь, но я слышу, как громко бьётся твоё сердце.",
            "Река течёт, и с ней уходят все мысли, что терзали.",
            "Туман скрывает горы вдали, но сердце видит больше.",
            "Пусть ветер треплет волосы, пусть свет мерцает вдалеке.",
            "Я слышу голос в тишине, он зовёт меня домой.",
            "Когда всё стихнет, ночь придёт, и тьма обнимет нас.",
            "Звёзды падают на землю, словно тихий шёпот сна.",
            "Душа поёт о вещах невидимых, что живут в каждом из нас.",
            "Огни Москвы горят вдали, они зовут, но я в тени.",
            "Природа вечна, и в её покое я нахожу себя.",
            "Кто знает, что скрывается за горизонтом в конце пути.",
            "Луна качается в реке, как лодка под ветром."
        ],
        "author": "Фёдор Тютчев"
    },
    {
        "fragment": [
            "Сжала руки под тёмной вуалью... 'Отчего ты сегодня бледна?' — Оттого, что я терпкой печалью Напоила его допьяна.",
            "Не с теми я, кто бросил землю На растерзание врагам. Их грубой лести я не внемлю, Им песен я своих не дам.",
            "Мне ни к чему одические рати И прелесть элегических затей. По мне, в стихах всё быть должно некстати, Не так, как у людей.",
            "Заплаканная осень, как вдова, В одеждах чёрных, всё сердца туманит... Перебирая мужнины слова, Она рыдать не перестанет.",
            "Весна пришла, но нет в душе тепла, лишь мрак.",
            "Я вспоминаю старые стихи, что пели мне о счастье.",
            "В ночной тиши, где снег блестит, я слышу тени прошлого.",
            "Мой дом забыт, но память греет сердце и зовёт.",
            "Не плачь, моя душа, весна вернётся снова.",
            "На берегах реки я вспоминаю детство, полное радости.",
            "Листья осени мне шепчут тайны давнего прошлого.",
            "В лунном свете я вижу тени, что поют мне песни.",
            "Куда уходит время, где его найти?",
            "Моя любовь, ты звезда в моём сердце.",
            "Когда наступит утро, ночь растает, как сон.",
            "Нас разлучают годы, но ты всегда со мной.",
            "Светлая грусть в глазах твоих, как зеркало души.",
            "Я слышу песни ветра, что зовут меня домой.",
            "Где кончается небо, там начинается путь к тебе.",
            "Река уносит меня, но память остаётся на берегу.",
            "Ты звезда моя, что светит в ночи.",
            "Сквозь туман я вижу свет, который зовёт меня.",
            "Осень снова приходит, принося с собой воспоминания.",
            "Где начинается весна, там я найду свою мечту."
        ],
        "author": "Анна Ахматова"
    },
    {
        "fragment": [
            "Ты жива ещё, моя старушка? Жив и я. Привет тебе, привет! Пусть струится над твоей избушкой Тот вечерний несказанный свет.",
            "До свиданья, друг мой, до свиданья. Милый мой, ты у меня в груди. Предназначенное расставанье Обещает встречу впереди.",
            "Гой ты, Русь, моя родная, Хаты — в ризах образа... Не видать конца и края — Только синь сосёт глаза.",
            "Не жалею, не зову, не плачу, Всё пройдёт, как с белых яблонь дым. Увяданья золотом охвачен, Я не буду больше молодым.",
            "В деревне тихо, слышен стук колёс вдали.",
            "Осенний лес горит, как золото, и манит взгляд.",
            "На речке утки крякают, а я мечтаю о свободе.",
            "Зима пришла, и лес укрыт белым покрывалом.",
            "Ветер в поле гуляет, как юный казак.",
            "На горке снег блестит, как алмазы в солнечном свете.",
            "Ты помнишь наши встречи, где звёзды светили нам?",
            "Ручей журчит, как песня, что звучит в сердце.",
            "Когда придёт весна, я снова буду счастлив.",
            "Над полями светит солнце, а я пою о Руси.",
            "Моя страна, ты как песня, что всегда звучит в душе.",
            "В дымке вечера я вижу звёзды, что зовут меня.",
            "На холмах леса шумят, как море во время бури.",
            "Ты моя радость, мой свет, моя звезда вечная.",
            "Сквозь ночной туман я вижу образ, что греет душу.",
            "Вокруг поля, леса и луга — всё поёт о родине.",
            "Ты мой друг, мой брат, мой светлый путь.",
            "На горизонте я вижу солнце, которое приветствует утро.",
            "Рассвет приходит, и я чувствую тепло нового дня.",
            "Ветер поёт, и я слышу его зов в сердце."
        ],
        "author": "Сергей Есенин"
    }
]


# Function to save fragments to a CSV file
def save_fragments_to_csv(data, filename="text_fragments.csv"):
    # Define column headers
    fieldnames = ["fragment", "author"]
    
    # Open the file in write mode
    with open(filename, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data
        for row in data:
            writer.writerow(row)
    
    print(f"Data saved to {filename}")

# Save the example data to a CSV file
save_fragments_to_csv(data)

# ========================================================================
# ==============        DATA PREPROCESSING      ===========================

# Step 1: Load the CSV File
def load_data(filename="text_fragments.csv"):
    data = pd.read_csv(filename)
    print(f"Data loaded successfully! Total records: {len(data)}")
    return data

# Step 2: Preprocess the Text
def preprocess_text(text):
    # Basic cleaning: Remove extra spaces and special characters
    text = text.strip()  # Remove leading/trailing spaces
    text = text.replace("\n", " ")  # Replace newlines with space
    return text

# # Step 3: Split the Data
# def split_data(data, test_size=0.2, val_size=0.1):
#     # Preprocess text
#     data["fragment"] = data["fragment"].apply(preprocess_text)
    
#     # Split into training + validation and test sets
#     train_val, test = train_test_split(data, test_size=test_size, stratify=data["author"], random_state=42)
    
#     # Split training + validation into separate training and validation sets
#     val_ratio = val_size / (1 - test_size)  # Adjust validation ratio
#     train, val = train_test_split(train_val, test_size=val_ratio, stratify=train_val["author"], random_state=42)
    
#     print(f"Data split completed:")
#     print(f" - Training set: {len(train)} samples")
#     print(f" - Validation set: {len(val)} samples")
#     print(f" - Test set: {len(test)} samples")
    
#     return train, val, test


def split_data(data, test_size=0.2, val_size=0.1):
    # Preprocess text
    data["fragment"] = data["fragment"].apply(preprocess_text)
    
    # Split into training + validation and test sets (no stratify)
    train_val, test = train_test_split(data, test_size=test_size, random_state=42)
    
    # Split training + validation into separate training and validation sets
    val_ratio = val_size / (1 - test_size)  # Adjust validation ratio
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=42)
    
    print(f"Data split completed:")
    print(f" - Training set: {len(train)} samples")
    print(f" - Validation set: {len(val)} samples")
    print(f" - Test set: {len(test)} samples")
    
    return train, val, test


# # Main Script
# if __name__ == "__main__":
#     # Load the data
#     data = load_data("text_fragments.csv")
    
#     # Split into train, validation, and test sets
#     train, val, test = split_data(data)
    
#     # Save the splits for later use
#     train.to_csv("train.csv", index=False, encoding="utf-8")
#     val.to_csv("val.csv", index=False, encoding="utf-8")
#     test.to_csv("test.csv", index=False, encoding="utf-8")
    
#     print("Train, validation, and test sets saved successfully!")

# ========================================================================
# ==============        TOKENIZATION & TEXT VECTORIZATION      ===========================

# step 1: load the pre-trained tokenizer
def load_tokenizer():
    tokenizer= BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    print('Tokenizer loaded successfully!\n')
    return tokenizer 

# step 2: tokenize the text data 
def tokenize_text(data, tokenize, max_length=128): 
    # tokenize and pad/truncate sequences to max_length 
    inputs= tokenize(
        list(data['fragment']), 
        padding='max_length' ,
        truncation=True, 
        max_length=max_length, 
        return_tensors='tf'
    )
    print(f"Tokenization completed! Tokens shape: {inputs['input_ids'].shape}")
    return inputs

# Convert tokenized inputs to tensorflow dataset
def create_tf_dataset(tokenized_inputs):
    dataset= tf.data.Dataset.from_tensor_slices({
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'token_type_ids': tokenized_inputs['token_type_ids'],
        })    
    print('Converted to TensorFlow Dataset successfully!\n')
    return dataset

# save tensorflow dataset 
def save_tf_dataset(dataset, directory):
    # ensure the directory exists 
    if not os.path.exists(directory):
        os.makedirs(directory)
    # save datasete to disk 
    tf.data.experimental.save(dataset, directory)
    print(f'Dataset saved to {directory} \n')

if __name__ == "__main__":
    # Load data
    train_data = pd.read_csv("train.csv")
    val_data = pd.read_csv("val.csv")
    test_data = pd.read_csv("test.csv")

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Tokenize datasets
    max_length = 128
    train_inputs = tokenize_text(train_data, tokenizer, max_length)
    val_inputs = tokenize_text(val_data, tokenizer, max_length)
    test_inputs = tokenize_text(test_data, tokenizer, max_length)

    # Convert tokenized inputs to TensorFlow datasets
    train_dataset = create_tf_dataset(train_inputs)
    val_dataset = create_tf_dataset(val_inputs)
    test_dataset = create_tf_dataset(test_inputs)

    # Save the datasets to directories
    save_tf_dataset(train_dataset, "train_inputs_dir")
    save_tf_dataset(val_dataset, "val_inputs_dir")
    save_tf_dataset(test_dataset, "test_inputs_dir")


    print("Tokenized inputs saved successfully!")