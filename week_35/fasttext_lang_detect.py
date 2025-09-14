import fasttext as ft

# Load the pretrained model
ft_model = ft.load_model("./pretrained_model/lid.176.bin")

def fasttext_language_predict(text, model = ft_model):

  text = text.replace('\n', " ")
  prediction = model.predict([text])

  return prediction

# Run the function for predictions
print(fasttext_language_predict(english_text))
print(fasttext_language_predict(french_text))