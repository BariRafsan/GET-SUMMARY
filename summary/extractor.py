#!pip install boilerpy3
from boilerpy3 import extractors

extractor = extractors.CanolaExtractor()

doc = extractor.get_doc_from_url('https://www.jugantor.com/international/611448/%E0%A6%A8%E0%A7%87%E0%A6%A6%E0%A6%BE%E0%A6%B0%E0%A6%B2%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%A8%E0%A7%8D%E0%A6%A1%E0%A6%B8%E0%A7%87%E0%A6%B0-%E0%A6%B0%E0%A6%BE%E0%A6%B7%E0%A7%8D%E0%A6%9F%E0%A7%8D%E0%A6%B0%E0%A6%A6%E0%A7%82%E0%A6%A4%E0%A6%95%E0%A7%87-%E0%A6%B0%E0%A7%81%E0%A6%B6-%E0%A6%AA%E0%A6%B0%E0%A6%B0%E0%A6%BE%E0%A6%B7%E0%A7%8D%E0%A6%9F%E0%A7%8D%E0%A6%B0-%E0%A6%AE%E0%A6%A8%E0%A7%8D%E0%A6%A4%E0%A7%8D%E0%A6%B0%E0%A6%A3%E0%A6%BE%E0%A6%B2%E0%A7%9F%E0%A7%87-%E0%A6%A4%E0%A6%B2%E0%A6%AC/')

page_title = doc.title
page_contents = doc.content

print(str(page_title), end = "\n\n")
print(str(page_contents))