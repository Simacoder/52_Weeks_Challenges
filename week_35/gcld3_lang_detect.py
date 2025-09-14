import gcld3

# First feature: Single Language detection
def cld3_single_language_detection(text):
    max_num_bytes = len(text)
    detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0,
                                            max_num_bytes=max_num_bytes)
    result = detector.FindLanguage(text=text)
    return {
        "language": result.language,
        "probability": result.probability
    }

# Second feature: Multiple Language detection
def cld3_multiple_language_detection(text, nb_language=2):
    max_num_bytes = len(text)
    detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0,
                                            max_num_bytes=max_num_bytes)
    languages = detector.FindTopNMostFreqLangs(text=text, num_langs=nb_language)
    return [{"language": l.language, "probability": l.probability} for l in languages]


# Example usage:
english_text = "This is a simple English sentence."
french_text = "Ceci est une phrase en français."

print(cld3_single_language_detection(english_text))
print(cld3_single_language_detection(french_text))
print(cld3_multiple_language_detection("Hello, bonjour, hola", nb_language=3))
