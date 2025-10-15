def text_analyzer(text, analysis_type="summary"):
    """
    Analyze text and provide various insights
    
    Args:
        text (str): Text to analyze
        analysis_type (str): Type of analysis to perform
    
    Returns:
        str: Analysis results
    """
    import re
    from collections import Counter
    
    if analysis_type == "summary":
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        
        return f"""Text Analysis Summary:
- Characters: {len(text)}
- Words: {len(words)}
- Sentences: {len([s for s in sentences if s.strip()])}
- Paragraphs: {len([p for p in paragraphs if p.strip()])}
- Average words per sentence: {len(words) / max(len([s for s in sentences if s.strip()]), 1):.1f}
"""
    
    elif analysis_type == "word_frequency":
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        top_words = word_freq.most_common(10)
        
        result = "Top 10 Most Frequent Words:\n"
        for word, count in top_words:
            result += f"- {word}: {count}\n"
        return result
    
    elif analysis_type == "readability":
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return "Cannot calculate readability: no sentences found"
        
        avg_sentence_length = len(words) / len(sentences)
        syllable_count = sum(count_syllables(word) for word in words)
        avg_syllables_per_word = syllable_count / max(len(words), 1)
        
        # Simple readability score (approximation)
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        if readability_score >= 90:
            level = "Very Easy"
        elif readability_score >= 80:
            level = "Easy"
        elif readability_score >= 70:
            level = "Fairly Easy"
        elif readability_score >= 60:
            level = "Standard"
        elif readability_score >= 50:
            level = "Fairly Difficult"
        elif readability_score >= 30:
            level = "Difficult"
        else:
            level = "Very Difficult"
        
        return f"""Readability Analysis:
- Average sentence length: {avg_sentence_length:.1f} words
- Average syllables per word: {avg_syllables_per_word:.1f}
- Readability score: {readability_score:.1f}
- Reading level: {level}
"""
    
    else:
        return f"Unknown analysis type: {analysis_type}"


def count_syllables(word):
    """Count syllables in a word (approximation)"""
    word = word.lower()
    vowels = "aeiouy"
    syllable_count = 0
    prev_was_vowel = False
    
    for char in word:
        if char in vowels:
            if not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = True
        else:
            prev_was_vowel = False
    
    # Handle silent e
    if word.endswith('e') and syllable_count > 1:
        syllable_count -= 1
    
    return max(syllable_count, 1)


def url_analyzer(url, analysis_type="basic"):
    """
    Analyze URLs and extract information
    
    Args:
        url (str): URL to analyze
        analysis_type (str): Type of analysis
    
    Returns:
        str: Analysis results
    """
    import re
    from urllib.parse import urlparse, parse_qs
    
    try:
        parsed = urlparse(url)
        
        if analysis_type == "basic":
            return f"""URL Analysis:
- Scheme: {parsed.scheme}
- Domain: {parsed.netloc}
- Path: {parsed.path}
- Query parameters: {len(parse_qs(parsed.query))}
- Fragment: {parsed.fragment or 'None'}
- Full URL length: {len(url)}
"""
        
        elif analysis_type == "security":
            issues = []
            
            if parsed.scheme != "https":
                issues.append("Not using HTTPS")
            
            if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', parsed.netloc):
                issues.append("Using IP address instead of domain")
            
            suspicious_patterns = [
                r'[0-9]{5,}',  # Long numbers
                r'[a-z]{20,}',  # Very long strings
                r'%[0-9a-f]{2}',  # URL encoding
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    issues.append(f"Suspicious pattern found: {pattern}")
            
            if not issues:
                issues.append("No obvious security issues detected")
            
            return "Security Analysis:\n" + "\n".join(f"- {issue}" for issue in issues)
        
        elif analysis_type == "params":
            query_params = parse_qs(parsed.query)
            if not query_params:
                return "No query parameters found"
            
            result = "Query Parameters:\n"
            for key, values in query_params.items():
                result += f"- {key}: {', '.join(values)}\n"
            return result
        
        else:
            return f"Unknown analysis type: {analysis_type}"
            
    except Exception as e:
        return f"Error analyzing URL: {str(e)}"
