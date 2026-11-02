"""Optional visual and content regressions; requires Playwright and a local browser.

Run with a Playwright-enabled Python: python -m unittest discover -s tests
-p test_story_browser.py. Windows uses installed Edge; other platforms use Chromium.
"""

import csv
import json
import sys
import unittest
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "python"
CHAPTERS = ("rule", "model", "proxy", "audit", "promotion", "appendix", "reproduce")


def table(name, folder=OUT):
    with open(folder / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row(rows, **match):
    found = [r for r in rows if all(r[k] == v for k, v in match.items())]
    assert len(found) == 1, (match, len(found))
    return found[0]


def fmt(value, digits):
    return f"{float(value):.{digits}f}".replace("-", "\u2212")


def pct(value, digits=1):
    return fmt(100 * float(value), digits) + "%"


@unittest.skipUnless(sync_playwright, "Optional Playwright dependency is not installed")
class StoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(
            channel="msedge" if sys.platform == "win32" else None
        )

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def tearDown(self):
        for context in self.browser.contexts:
            context.close()

    def open_story(self, width, motion="no-preference", height=1000):
        page = self.browser.new_page(
            viewport={"width": width, "height": height},
            reduced_motion=motion,
            color_scheme="light",
        )
        page.errors = []
        page.requests = []
        page.on("console", lambda m: m.type == "error" and page.errors.append(m.text))
        page.on("pageerror", lambda e: page.errors.append(str(e)))
        page.on("request", lambda r: page.requests.append(r.url))
        page.goto((ROOT / "docs" / "index.html").as_uri() + "?scoutTheme=light")
        return page

    def text(self, page, key):
        return page.locator(f'[data-k="{key}"]').first.inner_text()

    def test_liner_glides_once_and_wake_finishes(self):
        page = self.open_story(1600)
        liner = page.locator(".voyage-liner")
        first_x = liner.bounding_box()["x"]
        page.wait_for_timeout(4200)
        page.wait_for_function(
            "document.querySelector('.hero-voyage').getAnimations({subtree:true})"
            ".every(a => a.playState === 'finished')",
            timeout=6000,
        )
        self.assertGreater(liner.bounding_box()["x"], first_x + 10)
        self.assertTrue(
            page.locator(".hero-voyage").evaluate(
                """el => {
                    const animations = el.getAnimations({subtree:true});
                    return animations.length === 3 && animations.every(a =>
                        a.playState === "finished" &&
                        a.effect.getComputedTiming().iterations === 1 &&
                        a.effect.getComputedTiming().endTime <= 4000);
                }"""
            )
        )
        self.assertTrue(
            page.locator(".voyage-wake").evaluate_all(
                "els => els.every(el => parseFloat(getComputedStyle(el).strokeDashoffset) === 0)"
            )
        )

    def test_voyage_stays_clear_of_text_at_mobile_tablet_and_desktop_widths(self):
        for width in (320, 390, 768, 1050, 1280, 1600):
            with self.subTest(width=width):
                page = self.open_story(width)
                art = page.locator(".hero-voyage").bounding_box()
                for selector in ("h1", ".eyebrow", ".lead", ".hero > .aside"):
                    text = page.locator(selector).first.bounding_box()
                    overlaps = (
                        art["x"] < text["x"] + text["width"]
                        and art["x"] + art["width"] > text["x"]
                        and art["y"] < text["y"] + text["height"]
                        and art["y"] + art["height"] > text["y"]
                    )
                    self.assertFalse(overlaps, f"Voyage overlaps {selector}")
                self.assertFalse(
                    page.evaluate("document.documentElement.scrollWidth > innerWidth")
                )
                page.close()

    def test_reduced_motion_retains_static_art_and_theme_controls_still_work(self):
        page = self.open_story(390, "reduce")
        art = page.locator(".hero-voyage")
        self.assertEqual(art.get_attribute("aria-hidden"), "true")
        self.assertEqual(art.get_attribute("focusable"), "false")
        self.assertEqual(
            art.evaluate("el => el.getAnimations({subtree:true}).length"), 0
        )
        self.assertGreater(
            page.locator(".voyage-liner").evaluate(
                "el => Number(getComputedStyle(el).opacity)"
            ),
            0,
        )
        hull = page.locator(".liner-hull")
        original = hull.evaluate("el => getComputedStyle(el).fill")
        page.click("#theme")
        self.assertNotEqual(hull.evaluate("el => getComputedStyle(el).fill"), original)
        page.select_option("#language", "Python")
        self.assertIn("Python", page.locator("#model-language").inner_text())

    def test_reduced_motion_shows_every_chapter_without_animation(self):
        page = self.open_story(1280, "reduce")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(300)
        self.assertEqual(page.evaluate("document.getAnimations().length"), 0)
        self.assertFalse(page.evaluate("document.documentElement.classList.contains('js-reveal')"))
        self.assertFalse(page.locator("#seats").evaluate("el => el.classList.contains('seats-anim')"))
        self.assertTrue(
            page.evaluate(
                "[...document.querySelectorAll('.reveal, .seat:not(.empty)')]"
                ".every(el => Number(getComputedStyle(el).opacity) === 1)"
            )
        )

    def test_research_context_is_readable_before_the_first_chart(self):
        for width in (320, 390, 1600):
            with self.subTest(width=width):
                page = self.open_story(width, "reduce")
                bridge = page.locator("#people-analytics-bridge")
                bridge.scroll_into_view_if_needed()
                self.assertTrue(bridge.is_visible())
                title = page.locator("#people-analytics-title")
                self.assertEqual(title.evaluate("el => el.tagName"), "H2")
                self.assertEqual(title.inner_text(), "Predicting an outcome is not explaining it.")
                self.assertIn(
                    "The findings remain specific to this incomplete historical sample",
                    bridge.inner_text(),
                )
                self.assertNotIn("portfolio", bridge.inner_text().lower())
                bounds = bridge.bounding_box()
                chart = page.locator("#policy-chart").bounding_box()
                self.assertLess(bounds["y"] + bounds["height"], chart["y"])
                self.assertGreaterEqual(bounds["x"], 0)
                self.assertLessEqual(bounds["x"] + bounds["width"], width)
                self.assertFalse(
                    page.evaluate("document.documentElement.scrollWidth > innerWidth")
                )
                page.close()

    def test_hero_acknowledges_history_without_memorial_or_scoring_claims(self):
        page = self.open_story(1280, "reduce")
        aside = page.locator(".hero > .aside").inner_text()
        self.assertIn("1,500", aside)
        self.assertIn("people, families", aside)
        self.assertIn("not why any individual survived or died", aside)
        for wording in ("in memory of", "not a survival game", "scores no one"):
            self.assertNotIn(wording, aside.lower())
        self.assertIn("Patterns", page.locator("h1").inner_text())
        credit = json.loads((ROOT / "data" / "provenance.json").read_text(encoding="utf-8"))
        self.assertIn(credit["required_credit"], aside)
        self.assertIn(credit["required_credit"], page.locator("footer").inner_text())
        self.assertEqual(page.locator("input[type=text], input[type=number]").count(), 0)

    def test_narrative_separates_historical_prediction_from_synthetic_auditing(self):
        page = self.open_story(1280, "reduce")
        bridge = page.locator("#people-analytics-bridge").inner_text()
        self.assertIn("separate, explicitly synthetic promotion example", bridge)
        self.assertIn("not a conclusion about real employees or organizations", bridge)
        self.assertIn(
            "not establish calibration across the probability range",
            page.locator("#audit").inner_text(),
        )
        self.assertIn("not establish that the models are equivalent", page.locator("#model").inner_text())
        self.assertIn("not an identified policy effect", page.locator("#model").inner_text())
        body = page.locator("main").inner_text().lower()
        for wording in (
            "learns the rule", "learns the decision", "hr policies work the same way",
            "both groups are well calibrated", "if this were promotion data",
        ):
            self.assertNotIn(wording, body)
        self.assertEqual(page.locator(".hero-voyage .voyage-liner").count(), 1)
        self.assertEqual(page.locator("#deck .boat").count(), 20)
        self.assertEqual(page.errors, [])

    def test_every_chapter_renders_numbers_from_the_committed_outputs(self):
        policy, sides, missing = table("policy_groups.csv"), table("boat_side.csv"), table("missing_age.csv")
        comparison, gaps = table("model_comparison.csv"), table("proxy_gap.csv")
        recovery, audit = table("sex_recovery.csv"), table("fairness_audit.csv")
        impact, coefs = table("sim_impact.csv"), table("sim_coefficients.csv")
        summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
        synthetic = table("synthetic_promotions.csv", ROOT / "data")
        women = [r for r in synthetic if r["gender"] == "Woman"]
        auc = lambda m: row(comparison, model=m, metric="roc_auc")["mean_model"]
        expected = {
            "wc-all": pct(row(policy, group="Women and children", pclass="All")["rate"]),
            "men-all": pct(row(policy, group="Men", pclass="All")["rate"]),
            "wc-3": pct(row(policy, group="Women and children", pclass="3")["rate"]),
            "stb-share": pct(row(sides, side="Starboard")["adult_men_share"], 0),
            "port-share": pct(row(sides, side="Port")["adult_men_share"], 0),
            "stb-n": row(sides, side="Starboard")["n"],
            "miss-3": pct(row(missing, pclass="3")["missing_share"]),
            "auc-ref": fmt(auc("Policy + structure"), 3),
            "auc-policy": fmt(auc("Policy only"), 2),
            "auc-structure": fmt(auc("Structure only"), 2),
            "auc-rf": fmt(auc("Random forest"), 3),
            "gap-blind": pct(row(gaps, source="Sex-blind")["share_of_observed_gap"], 0),
            "gap-title": pct(row(gaps, source="Sex-blind + title")["share_of_observed_gap"], 0),
            "rec-without": fmt(row(recovery, features="Without title")["mean_auc"], 2),
            "rec-with": fmt(row(recovery, features="With title")["mean_auc"], 4),
            "ir-male": fmt(row(audit, grouping="Sex", group="Male")["impact_ratio"], 3),
            "lir-male": fmt(row(audit, grouping="Sex", group="Male")["label_impact_ratio"], 2),
            "hist-ir": fmt(row(impact, label="Historical decisions", group="Woman")["impact_ratio"], 2),
            "merit-ir": fmt(row(impact, label="Merit-only counterfactual", group="Woman")["impact_ratio"], 2),
            "hist-w": pct(row(impact, label="Historical decisions", group="Woman")["rate"]),
            "woman-est": fmt(row(coefs, model="Gender visible", term="woman")["estimate"], 2),
            "pt-blind": fmt(row(coefs, model="Gender-blind", term="part_time")["estimate"], 2),
            "pt-w": pct(sum(int(r["part_time"]) for r in women) / len(women)),
            "sim-auc": fmt(table("sim_recovery.csv")[0]["test_auc"], 2),
            "test-n": f"{summary['test_n']:,}",
            "train-n": f"{summary['train_n']:,}",
        }
        page = self.open_story(1440, "reduce")
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(self.text(page, key), value)
        for chapter in CHAPTERS:
            with self.subTest(chapter=chapter):
                section = page.locator(f"#{chapter}")
                self.assertEqual(section.locator("h2").count(), 1)
                self.assertTrue(section.locator("h2").is_visible())
        counts = {"#policy-chart": 4, "#child-chart": 4, "#model-chart": 6, "#proxy-chart": 3,
                  "#sim-impact-chart": 2, "#sim-coef-chart": 6, "#sim-gap-chart": 4, "#rates": 3}
        for selector, count in counts.items():
            with self.subTest(chart=selector):
                self.assertEqual(page.locator(f"{selector} .rc-row").count(), count)
        self.assertEqual(page.locator("#seats circle").count(), 1178)
        self.assertEqual(page.locator("#seats circle:not(.empty)").count(), 711)
        self.assertEqual(page.locator("#audit-grid figure").count(), 6)
        self.assertEqual(page.locator("#scores tr").count(), 3)
        self.assertIn(f"N = {summary['train_n']:,}", page.locator("#rule-status").inner_text())
        self.assertIn(f"holdout n = {summary['test_n']}", page.locator("#model-language").inner_text())
        self.assertEqual(page.errors, [])
        self.assertTrue(all(url.startswith(("file:", "data:")) for url in page.requests), page.requests)

    def test_controls_are_operable_and_small_cells_are_suppressed(self):
        page = self.open_story(1280, "reduce")
        cutoffs = table("child_cutoffs.csv")
        page.check("#cutoff input[value='12']")
        child = row(cutoffs, cutoff="12", pclass="All")
        self.assertIn(f"{int(child['n'])} children", page.locator("#child-status").inner_text())
        page.click("[data-metric='brier']")
        self.assertEqual(page.get_attribute("[data-metric='brier']", "aria-pressed"), "true")
        self.assertIn("Brier", page.locator("#model-chart").inner_text())
        page.focus("#tab-sex")
        page.keyboard.press("End")
        self.assertEqual(page.get_attribute("#tab-sexclass", "aria-selected"), "true")
        self.assertEqual(page.evaluate("document.activeElement.id"), "tab-sexclass")
        audit = [r for r in table("fairness_audit.csv") if r["grouping"] == "Sex x class"]
        suppressed = sum(1 for r in audit for m in ("tpr", "fpr", "ppv") if r[m] == "")
        self.assertGreater(suppressed, 0)
        self.assertEqual(page.locator("#audit-grid .rc-suppressed").count(), suppressed)
        self.assertIn("n<20", page.locator("#audit-grid").inner_text())
        page.keyboard.press("ArrowRight")
        self.assertEqual(page.get_attribute("#tab-sex", "aria-selected"), "true")
        page.fill("#confidence", "100")
        page.dispatch_event("#confidence", "input")
        self.assertIn("rules meet these filters", page.locator("#rule-status").inner_text())
        self.assertEqual(page.errors, [])

    def test_mobile_pages_have_no_horizontal_overflow(self):
        for width in (320, 390):
            with self.subTest(width=width):
                page = self.open_story(width, "reduce", height=800)
                page.click("#tab-sexclass")
                widest = page.evaluate(
                    """() => Math.max(...[...document.querySelectorAll('body *')]
                        .filter(el => !el.closest('.table-scroll, pre'))
                        .map(el => el.getBoundingClientRect().right))"""
                )
                self.assertLessEqual(widest, width + 1)
                self.assertFalse(
                    page.evaluate("document.documentElement.scrollWidth > innerWidth")
                )
                self.assertEqual(page.errors, [])
                page.close()

    def test_rule_evidence_uses_the_selected_outcome_and_exact_denominators(self):
        rules = table("rules.csv")
        page = self.open_story(1280, "reduce")
        for outcome in ("Outcome=Survived", "Outcome=Did not survive"):
            rule = next(r for r in rules if r["consequent"] == outcome)
            key = json.dumps([rule["antecedent"], rule["consequent"]], separators=(",", ":"))
            page.select_option("#rule-select", key)
            chart = page.locator("#rule-evidence-chart")
            text = chart.inner_text()
            self.assertIn(pct(rule["baseline"]), text)
            self.assertIn(pct(rule["confidence"]), text)
            self.assertIn(
                f"{int(rule['consequent_count']):,} of {int(rule['N']):,}", text
            )
            self.assertIn(
                f"{int(rule['count']):,} of {int(rule['antecedent_count']):,}", text
            )
            self.assertEqual(chart.locator(".rc-bar").count(), 2)
            widths = chart.locator(".rc-bar").evaluate_all(
                "els => els.map(el => parseFloat(el.style.width))"
            )
            self.assertAlmostEqual(widths[0], float(rule["baseline"]) * 100, delta=.0001)
            self.assertAlmostEqual(widths[1], float(rule["confidence"]) * 100, delta=.0001)
            reading = page.locator("#rule-evidence-reading").inner_text()
            self.assertIn(f"{float(rule['lift']):.2f}×", reading)
            self.assertIn("not an intervention effect", reading)
            self.assertIn(outcome.replace("Outcome=", ""), reading)
        self.assertIn("not independent groups", page.locator("#rule-evidence").inner_text())
        self.assertEqual(page.errors, [])

    def test_rule_evidence_handles_empty_filters_and_language_changes(self):
        page = self.open_story(390, "reduce")
        select = page.locator("#rule-select")
        select.focus()
        page.keyboard.press("End")
        page.keyboard.press("Enter")
        selected = select.input_value()
        self.assertNotEqual(selected, select.locator("option").first.get_attribute("value"))
        page.fill("#confidence", "100")
        page.dispatch_event("#confidence", "input")
        self.assertTrue(page.locator("#rule-empty").is_visible())
        self.assertTrue(select.is_disabled())
        self.assertTrue(page.locator("#rule-evidence").is_hidden())
        self.assertEqual(page.locator("#rule-evidence-chart").inner_text(), "")
        page.fill("#confidence", "60")
        page.dispatch_event("#confidence", "input")
        self.assertTrue(page.locator("#rule-evidence").is_visible())
        self.assertFalse(select.is_disabled())
        if page.locator("#language option[value=R]").is_enabled():
            page.select_option("#language", "R")
            key = json.loads(select.input_value())
            rule = row(table("rules.csv", ROOT / "outputs" / "r"),
                       antecedent=key[0], consequent=key[1])
            self.assertIn(pct(rule["confidence"]), page.locator("#rule-evidence-chart").inner_text())
        self.assertEqual(page.errors, [])

    def test_model_focus_links_views_and_preserves_every_comparison(self):
        page = self.open_story(1280, "reduce")
        original_scores = page.locator("#scores").inner_text()
        original_paths = page.locator("#roc .curve, #calibration .curve").evaluate_all(
            "els => els.map(el => el.getAttribute('d'))"
        )
        select = page.locator("#model-focus")
        select.focus()
        page.keyboard.press("End")
        page.keyboard.press("Enter")
        self.assertEqual(select.input_value(), "Random forest")
        for model in ("Random forest", "Bayesian logistic", "Prevalence baseline"):
            page.select_option("#model-focus", model)
            for container in ("#scores", "#roc", "#calibration", "#model-legend"):
                focused = page.locator(f"{container} .is-focused")
                self.assertEqual(focused.count(), 1)
                self.assertEqual(focused.get_attribute("data-model"), model)
                self.assertEqual(page.locator(f"{container} [data-model]").count(), 3)
            self.assertIn("not select a winner", page.locator("#model-focus-status").inner_text())
        if page.locator("#language option[value=R]").is_enabled():
            page.select_option("#language", "R")
            self.assertEqual(
                page.locator("#roc .is-focused").get_attribute("data-model"),
                "Prevalence baseline",
            )
            page.select_option("#language", "Python")
        page.select_option("#model-focus", "")
        self.assertEqual(page.locator(".model-series.is-context, #scores .is-focused").count(), 0)
        self.assertEqual(page.locator("#scores").inner_text(), original_scores)
        self.assertEqual(
            page.locator("#roc .curve, #calibration .curve").evaluate_all(
                "els => els.map(el => el.getAttribute('d'))"
            ), original_paths,
        )
        page.click("#theme")
        self.assertEqual(page.locator("#model-focus").input_value(), "")
        self.assertEqual(page.errors, [])


if __name__ == "__main__":
    unittest.main()
