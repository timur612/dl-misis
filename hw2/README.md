# hw2

## Decoding Methods Overview

При использовании стохастических декодеров результаты будут рандомными при каждом запуске, а детерминированные всегда возвращают одни и те же токены.

Общие проблемы:
- Сказка:  
  • Стохастика даёт разнообразие, но может терять сюжет и уходить в «галлюцинации».  
  • Детeрminiрованность часто ведёт к повторениям и «шаблонности».  
- JSON:  
  • Стохастика рискует нарушить строгий формат (пропустить кавычки, поставить лишнюю запятую).  
  • Детeрminiрованность может застревать на неудачных шаблонах.

### 1. sampling_decode (без температуры)

- Тип: стохастический  
- Алгоритм: выбор токена пропорционально softmax(logits)  
- Проблемы: высокая вариативность, возможный уход в бессвязный текст  

Пример сказки:
```
Once upon a time, in a quiet clearing on the edge of the great spruce forest, there lived the tiniest hedgehog you could imagine. His name was Sonic, and though he was small, he carried in his little heart an unbounded curiosity. Every morning he’d scurry between mushrooms and ferns, humming happily as the dew sparkled on his quills. One day, drawn by a distant melody, Sonic discovered an old storyteller owl perched atop an ancient oak. From that moment on, the hedgehog joined the owl’s nightly tales, and together they wove new legends beneath the silver glow of the moon.
```
Пример JSON:
```json
{"contractor":"Mike","sum":100.50,"currency":"RUB"}
```

### 2. sampling_decode with temperature

- Тип: стохастический  
- Алгоритм: softmax(logits/temperature) — регулирует «остроту» распределения  
- Проблемы:  
  • Низкая температура → слишком консервативно, часто повторения.  
  • Высокая температура → ещё больше бессвязности.

### 3. nucleus_decode (top-p)

- Тип: стохастический  
- Алгоритм: выбирает из минимального набора токенов с суммарной вероятностью ≥ top_p  
- Проблемы: схожа с обычным sampling, но лучше балансирует «длинный хвост» распределения.

### 4. greedy_decode

- Тип: детерминированный  
- Алгоритм: выбирает токен с максимальным логитом на каждом шаге  
- Проблемы:  
  • Часто застревает в локальных пиках, генерация скучная и повторяющаяся.

### 5. beam_search_decode

- Тип: детерминированный  
- Алгоритм: хранит топ-K кандидатов, расширяет их и выбирает лучшую последовательность с учётом length_penalty  
- Проблемы:  
  • При малом beam_width — тупиковые фрагменты.  
  • При большом — медленно и тоже может повторяться.

