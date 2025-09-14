def LanguageDetector():
    raise NotImplementedError

def spacy_language_detection(text, model):

  pipeline = list(dict(model.pipeline).keys())

  if(not "language_detector" in pipeline):
    model.add_pipe(LanguageDetector(), name = "language_detector", last=True)
    
  doc = model(text)

  return doc._.language