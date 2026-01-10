# Movie Recommendation

## Цель проекта
Разработать алгоритм для рекомендации фильмов на основе истории взаимодействий пользователя.
- В качестве метрик для оценки качества рекомендаций используются Recall@1, Recall@10, NDCG@10. Цель — достичь NDCG@10 выше 0.1.
- Для бесшовного взаимодействия пользователя с сервисом время обработки пользователя не должно превышать 1000 мс.

---

## Набор данных
- Используется открытый датасет MovieLens-1m, содержащий более $10^6$ взаимодействий пользователя с фильмами.
- Для рекомендации на основе истории пользователя берется только информация о взаимодействии с фильмами.
- Данные были очищены от шумных взаимодействий (пользователей, оценивших менее 5 фильмов).
- Валидационные метки — предпоследние взаимодействия случайных 512 пользователей.
- Данные для тестирования — последние взаимодействия всех пользователей.

---

## План экспериментов
1. Baseline модель — SASRec.  
2. Подбор гиперпараметров.  
3. Калибровка оценки вероятности взаимодействия пользователя с фильмом.  
4. Оценка по метрикам Recall@1, Recall@10, NDCG@10.  

---

## Запуск проекта
Склонируйте репозиторий и перейдите в его корневую директорию.

### Установка зависимостей
```bash
pip install -r requirements.txt
pip install -e .
```

### Данные для обучения и чекпоинты моделей
Версионирование данных и моделей машинного обучения производится с помощью инструмента DVC. Сами данные и модели находятся на удаленном хранилище [Google Drive](https://drive.google.com/drive/folders/16f4rf9NXtGX4yh2KIUgndEgae26Nh_JQ?usp=drive_link). Чтобы получить к ним доступ, необходимо последовать [инструкции](https://github.com/treeverse/dvc/issues/10516#issuecomment-2289652067).

Необходимо добавить файл config.local в директории .dvc, используя значения, полученные после прохождения инструкции, указанной выше
```
['remote "storage"']
    gdrive_client_id = <YOUR CLIENT ID>
    gdrive_client_secret = <YOUR SECRET>
```

Чтобы подтянуть данные из удаленного хранилища выполните команду
``` bash
dvc pull
```

### Предобработка данных
Данный шаг, можно пропустить, так как предобработанные данные тоже находятся на удаленном хранилище.
Предобработка данных выполняется с помощью следующей команды:

```
python datasets/MovieLens_Large/preprocess_ml1m.py
```
которая из сырых данных сформирует датасеты для обучения, валидации и теста.

### Обучение модели
```bash
python scripts/train.py --config=recsys_project/configs/config_sasrec.py --config_hyper=recsys_project/configs/config_training.py
```

В качестве ***config*** и ***config_hyper*** возможно передать свою конфигурацию модели и гиперпараметры обучения соответственно.

После запуска обучения автоматически создастся директория `mlruns`, в которой сохраняюся метрики во время обучения. Для просмотра логгов необходимо выполнить команду

```
mlflow server --port 5012
```
Также можно указать другой порт.

### Оценка качества
```bash
python scripts/evaluate.py --config=recsys_project/configs/config_sasrec.py --config_hyper=recsys_project/configs/config_training.py --checkpoint=checkpoints/best_model.pt
```
В качестве чекпоинта можно также передать свою сохраненную модель с соответствующей конфигурацией.

### Docker
Получить Docker образ можно скачав его с [Dockerhub](https://hub.docker.com/r/lamcrbt/ml-app)

```
docker pull lamcrbt/ml-app:v1
```
Или собрав его вручную с помощью Dockerfile проекта
```
docker build -t ml-app:v1 .
```

Чтобы запустить образ, скачанный с Dockerhub выполните команду
```
docker run --rm -v $(pwd)/csv:/data  lamcrbt/ml-app:v1 \
    --top_k=10 \
    --input_path=/data/input.csv --output_path=/data/output.csv 
```

Если собрали docker образ сами
```
docker run --rm -v $(pwd)/csv:/data  ml-app:v1 \
    --top_k=10 \
    --input_path=/data/input.csv --output_path=/data/output.csv 
```
В качестве параметров можно указать:
* --top_k - количество предсказаний для каждого пользователя
* --input_path=/data/input.csv - путь, в котором хранятся входные данные модели (на каждой строчке находится последовательность взаимодействий пользователей, разделенных ",". Например, 968,962,161,199,2003)
* --output_path=/data/output.csv - путь до csv-файла с top-k предсказаниями для каждого пользователя. Имеет такой же формат как и у входных данных.
* Необходимо пробросить корректный путь в docker. -v PATH_TO_INPUT_FOLDER:/data

### TorchServe
Создайте директорию, если ее еще нет, где будет храниться .mar архив.
```bash
mkdir -p torchserve/model-store
```
Затем соберите .mar архив, в котором будет находиться информация о модели, необходимая для инициализации в handler.py на сервере
```
torch-model-archiver   --model-name sasrec   --version 1.0 \
    --serialized-file checkpoints/best_model.pt   --handler torchserve/handler.py \
    --extra-files "recsys_project/configs/config_sasrec.py,datasets/MovieLens_Large/dataset_stats.json" \
    --export-path torchserve/model-store --force
```

Чтобы собрать docker, выполните следующую команду
```
docker build -f torchserve/Dockerfile.torchserve -t sasrec-serve:v1 .
```

Эти шаги можно пропустить, загрузив Docker из [Dockerhub](https://hub.docker.com/r/lamcrbt/sasrec-serve). В этом случае далее следует использовать `lamcrbt/sasrec-serve:v1` вместо `sasrec-serve:v1`
```
docker pull lamcrbt/sasrec-serve:v1
```

Для запуска докера отнаследованного от torchserve выполните. (Выставите другие порты, если эти заняты)
```
docker run -d -p 8068:8080 -p 8069:8081 sasrec-serve:v1
```

Чтобы отправить свой запрос выполните
```
curl -X POST http://localhost:8068/predictions/sasrec \
    -H "Content-Type: application/json" \
    --data-binary @<PATH_TO_JSON_FOLDER>/input.json
```

Можно также отправить запрос в виде строки.
```
curl -X POST http://localhost:8068/predictions/sasrec \
    -H "Content-Type: application/json" \
    --data-binary '{"item_sequence": [560,280,9,1464,493,227,229]}'
```

На входе ожидается json файл следующего вида:
```json
{
    "item_sequence": [560,280,9,1464,493,227,229]
}
```
По ключу `item_sequence` находится список взаимодействий пользователя. На выходе появится тоже json-файл с ключем `recommendations`. В нем находится список из 10-ти рекомендуемых моделью айтемов.

```json
{
    "recommendations": [1,2,3,4,5,6,7,8,9,10]
}
```


## Текущие метрики
| Model  | Loss | Recall@1 | Recall@10 | NDCG@10 |
|--------|------|----------|-----------|---------|
| SASRec | BCE  | 0.0265   | 0.1505    | 0.0777  |
