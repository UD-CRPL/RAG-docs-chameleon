import re
import unittest

def sanitization(text:str)-> str:
    patterns= {}

    for name, pattern in patterns.item():
        text = re.sub(patter, f"[REDUCTED_{name.upper()}]", text)

    return text 


class testfunc(unittest.TestCase):
    def test1(self):
        input = ""
        output = ""
        self.assertEqual(sanitization(text), output)

    def test1(self):
        input = ""
        output = ""
        self.assertEqual(sanitization(text), output)

    def test1(self):
        input = ""
        output = ""
        self.assertEqual(sanitization(text), output)

    def test1(self):
        input = ""
        output = ""
        self.assertEqual(sanitization(text), output)

    def test1(self):
        input = ""
        output = ""
        self.assertEqual(sanitization(text), output)

    def test1(self):
        input = ""
        output = ""
        self.assertEqual(sanitization(text), output)



if __name__ == "__main__":
    unittest.main()


