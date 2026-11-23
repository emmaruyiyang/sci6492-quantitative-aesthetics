#cleans the two raw texts into plain prose ready for word2vec training
#output format: one paragraph per line, no headings, contents, footnotes, captions or tables
#
#sources:
#  raw/cano_raw.txt  : The Complete Sherlock Holmes, https://sherlock-holm.es/stories/plain-text/cano.txt
#  raw/abyss_raw.txt : The People of the Abyss (Jack London), https://www.gutenberg.org/ebooks/1688

import os
import re

this_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(this_dir, 'raw')


#shared helpers___________________________________________________________

def normalize_characters(text):
    #curly quotes -> straight quotes, so "don’t" and "don't" become the same token
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('“', '"').replace('”', '"')
    #the double hyphen "--" is the ascii form of an em dash; use one form for both texts
    text = text.replace('--', '—')
    return text


def join_paragraphs(lines):
    #lines are hard wrapped at ~70 characters, so join them back into paragraphs
    #(blank lines separate paragraphs)
    paragraphs = []
    current = []
    for line in lines:
        if line.strip() == '':
            if current:
                paragraphs.append(' '.join(current))
                current = []
        else:
            current.append(line.strip())
    if current:
        paragraphs.append(' '.join(current))

    #tidy whitespace inside each paragraph
    return [re.sub(r'\s+', ' ', p).strip() for p in paragraphs]


def write_paragraphs(paragraphs, file_name):
    path = os.path.join(this_dir, file_name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(paragraphs) + '\n')
    words = sum(len(p.split()) for p in paragraphs)
    print(f'{file_name}: {len(paragraphs)} paragraphs, {words} words')


#Sherlock Holmes__________________________________________________________

def clean_sherlock():
    with open(os.path.join(raw_dir, 'cano_raw.txt'), encoding='utf-8') as f:
        lines = f.read().split('\n')

    #drop the closing "provided as-is" notice after the last separator
    last_separator = max(i for i, l in enumerate(lines) if l.strip() == '----------')
    lines = lines[:last_separator]

    body = []
    skipping_block = False   #inside a footnote or a picture caption
    for line in lines:
        stripped = line.strip()

        #body text is indented by exactly 5 spaces; titles, tables of contents,
        #chapter headings and cipher drawings are indented further, so drop those
        if stripped and not re.match(r'^ {5}\S', line):
            body.append('')
            continue

        #footnote block: starts with a "-----" rule, ends at a blank line
        if stripped == '-----':
            skipping_block = True
            continue
        #picture caption: "[ Picture: ... ]", sometimes over two lines
        if stripped.startswith('[ Picture:'):
            skipping_block = not stripped.endswith(']')
            continue
        if skipping_block:
            if stripped == '' or stripped.endswith(']'):
                skipping_block = False
            continue

        body.append(line)

    paragraphs = join_paragraphs(body)

    cleaned = []
    for p in paragraphs:
        p = normalize_characters(p)
        p = re.sub(r'\*\d+', '', p)             #footnote markers like "*1"
        p = re.sub(r'\[([^\]]*)\]', r'\1', p)   #keep bracketed asides, drop the brackets
        p = re.sub(r'\s+', ' ', p).strip()
        #drop signatures and initials that stand alone, like "S. H." or "F. M."
        if re.fullmatch(r'[A-Z. ]{1,8}', p):
            continue
        if p:
            cleaned.append(p)

    write_paragraphs(cleaned, 'sherlock_holmes_clean.txt')


#The People of the Abyss__________________________________________________

def clean_abyss():
    with open(os.path.join(raw_dir, 'abyss_raw.txt'), encoding='utf-8') as f:
        text = f.read().replace('\r\n', '\n')

    #keep only what lies between the project gutenberg start and end markers
    start = re.search(r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*', text)
    end = re.search(r'\*\*\* END OF THE PROJECT GUTENBERG EBOOK', text)
    text = text[start.end():end.start()]

    #the book proper starts at the epigraph that follows the table of contents
    text = text[text.index('The chief priests and rulers cry'):]

    #fix a typo in the gutenberg edition
    text = text.replace('vLies dead upon the plain', 'Lies dead upon the plain')

    lines = text.split('\n')
    body = []
    skip_next_heading = False
    skipping_footnote = False
    for line in lines:
        stripped = line.strip()

        #chapter headings: "CHAPTER I." followed by the chapter title on the next line
        if re.fullmatch(r'CHAPTER [IVXLC]+\.', stripped):
            skip_next_heading = True
            body.append('')
            continue
        if skip_next_heading and stripped:
            skip_next_heading = False
            body.append('')
            continue

        #footnote bodies start with " [1]" and run until a blank line
        if re.match(r'^\s*\[\d+\]', line):
            skipping_footnote = True
            continue
        if skipping_footnote:
            if stripped == '':
                skipping_footnote = False
            continue

        #whole lines in capitals: "PREFACE", signatures like "JACK LONDON.",
        #places like "PIEDMONT, CALIFORNIA.", newspaper headlines
        if stripped and re.fullmatch(r"[A-Z .,'\-—]+", stripped):
            body.append('')
            continue

        #section breaks like "* * * * *"
        if stripped and re.fullmatch(r'[* ]+', stripped):
            body.append('')
            continue

        #rows of the statistical tables: text followed by columns of numbers
        #(one table is tab separated, and tabs appear nowhere else in the book)
        if '\t' in line or re.search(r'\S {2,}[\$\d]', stripped) or re.search(r'\d+/\d+\s*$', stripped):
            body.append('')
            continue

        body.append(line)

    paragraphs = join_paragraphs(body)

    cleaned = []
    for p in paragraphs:
        p = normalize_characters(p)
        p = re.sub(r'\[\d+\]', '', p)           #footnote markers like "[1]"
        p = re.sub(r'\[([^\]]*)\]', r'\1', p)   #keep glosses like [food], drop the brackets
        p = p.replace('_', '')                  #_underscores_ mark italics
        p = re.sub(r'\s+', ' ', p).strip()
        #lines that only introduce a table that is now gone, e.g. "Out of 81,951 deaths in London (1884):—"
        if p.endswith(':—') and len(p.split()) < 12 and re.search(r'\d', p):
            continue
        if p:
            cleaned.append(p)

    write_paragraphs(cleaned, 'people_of_the_abyss_clean.txt')


clean_sherlock()
clean_abyss()
