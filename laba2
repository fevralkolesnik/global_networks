from crc64iso.crc64iso import crc64
from math import ceil, log2
from random import randint

#pip install crc64iso

text = """За последние несколько лет во многих областях deep learning (например, в обработке естественного языка) стала популярной идея self-supervised learning. Оказалось, что для получения полезных в целевой задаче представлений не нужна большая размеченная выборка. Достаточно обучить модель на какой-то простой сигнал, построенный из неразмеченных данных, которых чаще всего в достатке. В частности, известные многим архитектуры BERT, GPT и языковая модель YaLM обучаются именно так.
На такие модели, однако, приходится тратить много ресурсов: обучение даже оригинального BERT-large занимало несколько дней на 64 TPU, а новые модификации и более крупные модели часто требуют ещё больше вычислений. Обучение BERT оценивается в 7 тысяч долларов, а в случае с моделями вроде GPT-3 сумма может достигать 12 миллионов долларов — терпимо для больших корпораций, но совершенно неподъёмно для энтузиастов и многих исследователей. При этом не для всех задач существуют качественные предобученные модели: например, из-за разнообразия мировых языков и доменов в NLP существует много областей, где специализированные нейросети по-прежнему работают лучше.
На помощь приходит парадигма добровольных вычислений (volunteer computing). Объединив мощности с другими энтузиастами, можно создать распределённую сеть, узлы которой будут совместно решать одну задачу. Такой подход давно и успешно используется в ряде научных областей, например, в моделировании свёртывания белка или поиске внеземных цивилизаций, но стандартные методы распределённого обучения нейросетей здесь плохо применимы. Нужны технологии, которые умеют работать с нестабильным участием каждого компьютера, разнородным «железом» и варьирующейся скоростью интернет-соединения.
Специально для решения этих проблем мы разработали метод децентрализованного обучения Distributed Deep Learning in Open Collaborations (или DeDLOC), подробно описанный в препринте на ArXiv. Вместе с энтузиастами мы смогли обучить близкую к state-of-the-art модель для бенгальского языка (шестого в мире по числу носителей) на дешёвых и даже бесплатных ресурсах, не прибегая к использованию GPU-кластера. Модель и библиотека для децентрализованного обучения нейросетей находятся в открытом доступе (ссылки — в конце поста), поэтому можете применить этот подход к интересным вам задачам уже сейчас."""

word_length = 68

def checksum(text):
  return crc64(text)
  
def text_to_bits(text, encoding='utf-8', errors="ignore"):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def bits_to_text(bits, encoding='utf-8', errors="ignore"):
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

def calcRedundantBits(m):
	for i in range(m):
		if(2**i >= m + i + 1):
			return i

def posRedundantBits(data, r):
	j = 0
	k = 1
	m = len(data)
	res = ''
	for i in range(1, m + r+1):
		if(i == 2**j):
			res = res + '0'
			j += 1
		else:
			res = res + data[-1 * k]
			k += 1
	return res

def calcParityBits(arr, r):
	n = len(arr)
	for i in range(r):
		val = 0
		for j in range(1, n + 1):
			if(j & (2**i) == (2**i)):
				val = val ^ int(arr[-1 * j])
		arr = arr[:n-(2**i)] + str(val) + arr[n-(2**i)+1:]
	return arr

def detectError(arr, nr):
	n = len(arr)
	res = 0
	for i in range(nr):
		val = 0
		for j in range(1, n + 1):
			if(j & (2**i) == (2**i)):
				val = val ^ int(arr[-1 * j])
		res = res + val*(10**i)
	return int(str(res), 2)

def remove_redundant_bits(data):
  r_idx = 1
  i_to_remove = []

  while r_idx+1 < len(data):
    i_to_remove.append(r_idx-1)
    r_idx *= 2

  i_to_remove = i_to_remove[::-1]

  for i in i_to_remove:
    data = data[:i] + data[i+1:]
  return data
  
def encode_word(data):
  length = len(data)
  red_count = calcRedundantBits(length)
  data_with_zeros = posRedundantBits(data[::-1], red_count)
  data_with_pars = calcParityBits(data_with_zeros[::-1], red_count)
  return data_with_pars[::-1]

def decode_word(word_data):
  return remove_redundant_bits(word_data)

def decode_word_error(word_data):
  decoded_word_data = remove_redundant_bits(word_data)
  r = calcRedundantBits(len(decoded_word_data))
  idx_error = detectError(word_data[::-1], r) - 1
  if idx_error == -1 or idx_error >= len(word_data):
    return decoded_word_data, 0
  word_data = word_data[:idx_error] + ("0" if word_data[idx_error] == "1" else "1") + word_data[idx_error+1:]
  return remove_redundant_bits(word_data), 1
  
def encode(text, word_length):
  data = text_to_bits(text)
  data_length = len(data)
  num_blocks = ceil(data_length / word_length)
  data_blocks = [
      encode_word(data[i*word_length:(i+1)*word_length])
      for i in range(num_blocks)
  ]
  return data_blocks, checksum(text)

def decode(words_data):
  decoded_data = "".join(
      decode_word(word_data)
      for word_data in words_data
  )
  decoded_text = bits_to_text(decoded_data)
  return checksum(decoded_text)

def decode_error(words_data):
  decoded_data = ""
  errors = 0
  for word_data in words_data:
    decoded_word, error = decode_word_error(word_data)
    decoded_data += decoded_word
    errors += error
  decoded_text = bits_to_text(decoded_data)
  return errors, checksum(decoded_text)
  
def place_error(words_data, word_num, bit_num):
  word_data = words_data[word_num]
  word_data = word_data[:bit_num] + ("0" if word_data[bit_num] == "1" else "1") + word_data[bit_num+1:]
  words_data[word_num] = word_data
  return words_data

def place_random_errors(words_data, max_errors_per_word):
  broken_words = 0
  for word_num in range(len(words_data)):
    amount_of_errors  = randint(0, max_errors_per_word)
    broken_words += 1 if amount_of_errors else 0
    for err in range(amount_of_errors):
      error_idx = randint(0, len(words_data[word_num])-1)
      words_data = place_error(words_data, word_num, error_idx)
  return words_data, broken_words
  
def main(text, word_length, max_broken_bits_per_word):
  encoded_blocks, check_init = encode(text, word_length)

  encoded_blocks, broken_words = place_random_errors(encoded_blocks, max_broken_bits_per_word)
  check_dec = decode(encoded_blocks)
  found_error_words, check_err = decode_error(encoded_blocks)

  print(f"Количество слов с ошибками: {broken_words}")
  print(f"Найдено ошибок в {found_error_words}")
  print(f"Конрольная сумма: {check_init == check_dec}")
  print(f"Контрольная сумма с исправлением ошибок: {check_init == check_err}")

main(text, word_length, max_broken_bits_per_word=0)
main(text, word_length, max_broken_bits_per_word=1)
main(text, word_length, max_broken_bits_per_word=2)
