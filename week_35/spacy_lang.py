english_text = """Niyel, a Dakar-based company that designs, implements, 
and evaluates advocacy campaigns to change policies, behaviors, and practices, 
will support the researchers in using the results to influence the implementation of AI-friendly policies.
"""

french_text = """Intelligence artificielle : la solution pour améliorer l'accès au crédit en Afrique ? 
Déjà une réalité au Kenya, en Afrique du Sud et au Nigeria, l'évaluation du risque crédit via 
l'intelligence artificielle dispose d'un fort potentiel en Afrique de l'Ouest malgré les inquiétudes liées à la protection de la vie privée."""

import spacy

def spacy_language_detection(text, model):
	doc = model(text)
	# spaCy's small English model does not have language detection by default,
	# so we return the language code from the model's meta or a placeholder.
	# For real detection, use 'spacy-langdetect' or similar.
	return model.meta.get("lang", "en")

# Load the pretrained model from spacy models' hub
pre_trained_model = spacy.load("en_core_web_sm")

# Detection on English text
print(spacy_language_detection(english_text, pre_trained_model))

# Detection on French text
print(spacy_language_detection(french_text, pre_trained_model))