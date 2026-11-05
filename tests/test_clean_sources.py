import unittest
from scripts.clean_evidence_sources import clean_body
class CleanSources(unittest.TestCase):
 def test_navigation_and_footer_removed(self):
  self.assertEqual(clean_body('导航\n## 重要事实\n甲\n## 概述\n乙\n## 世卫组织的应对\n机构信息\n页脚'),'## 重要事实\n甲\n## 概述\n乙')
 def test_missing_boundaries_rejected(self):
  with self.assertRaises(ValueError):clean_body('导航\n正文')
if __name__=='__main__':unittest.main()
