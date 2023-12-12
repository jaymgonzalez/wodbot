import unittest
from datetime import datetime, timedelta
from scrape_tetuan import reverse_date_range, get_previous_month_first_day

class TestScrapeTetuan(unittest.TestCase):
    def test_reverse_date_range(self):
        start_date = "2021-01-01"
        end_date = "2020-12-01"
        expected_output = [
            "2021-01-01",
            "2021-01-02",
            "2021-01-03",
            "2020-12-02",
            "2020-12-01"
        ]
        output = reverse_date_range(start_date, end_date)
        self.assertEqual(output, expected_output)

    def test_get_previous_month_first_day(self):
        date = datetime(2021, 1, 15)
        expected_output = datetime(2020, 12, 1)
        output = get_previous_month_first_day(date)
        self.assertEqual(output, expected_output)

if __name__ == "__main__":
    unittest.main()