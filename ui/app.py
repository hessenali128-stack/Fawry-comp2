"""
app.py
------
Entry point. Assembles the reusable pieces from theme.py / components.py /
pages.py into one Fawry-styled Gradio app with:
  - Sign up
  - Login
  - Home (dashboard)
  - Wallet
  - Offers
  - Quick pay
Navigation is done client-side-style by toggling the `visible` flag of
each page's Column, driven by gr.State + simple button callbacks -- no
page reload, just like the reference mobile UI.

Run with:  python app.py
"""

import gradio as gr

from theme import CUSTOM_CSS
import components as C
import pages as P

APP_PAGES = ["home", "wallet", "offers", "quickpay"]


FRONTEND_JS = """
        const API = "/api";
        function fwUserId() {
          const saved = localStorage.getItem("fawry_user_id");
          if (saved) return saved;
          const a = document.querySelector("#fw-login-id input");
          const b = document.querySelector("#fw-signup-id input");
          return (a && a.value) || (b && b.value) || "guest";
        }
        async function fwLoadOffers() {
          const row = document.querySelector(".fw-foryou-row");
          if (!row) return;
          const uid = fwUserId();
          if (!uid || uid === "guest") return;
          if (row.dataset.fwUserId === uid && row.children.length) return;
          try {
            const r = await fetch(`${API}/recommendations/${encodeURIComponent(uid)}`);
            const data = await r.json();
            row.innerHTML = (data.recommendations || []).map((x,i) => `
              <div class="fw-foryou-card" data-offer-id="${x.offer_id}">
                <div class="fw-cashback-icon">🎁</div>
                <div>
                  <div class="fw-cashback-title">${i+1}. ${x.title}</div>
                  <div class="fw-cashback-sub">${x.subtitle}</div>
                </div>
              </div>`).join("");
            row.dataset.fwUserId = uid;
          } catch (e) {}
        }
        document.addEventListener("click", async (e) => {
          const card = e.target.closest(".fw-foryou-card[data-offer-id]");
          if (!card) return;
          const uid = fwUserId();
          if (!uid || uid === "guest") return;
          card.style.opacity = "0.65";
          try {
            const r = await fetch(`${API}/offers/click`, {
              method: "POST", headers: {"Content-Type":"application/json"},
              body: JSON.stringify({user_id: uid, offer_id: card.dataset.offerId})
            });
            const data = await r.json();
            const row = document.querySelector(".fw-foryou-row");
            if (row && data.recommendations) {
              row.innerHTML = data.recommendations.map((x,i) => `
                <div class="fw-foryou-card" data-offer-id="${x.offer_id}">
                  <div class="fw-cashback-icon">🎁</div><div>
                  <div class="fw-cashback-title">${i+1}. ${x.title}</div>
                  <div class="fw-cashback-sub">${x.subtitle}</div></div>
                </div>`).join("");
              row.dataset.fwUserId = uid;
            }
          } catch (err) {}
        });
        document.addEventListener("click", (e) => {
          const text = (e.target.innerText || "").trim();
          if (text === "Log in" || text === "Create account") {
            setTimeout(() => {
              const a = document.querySelector("#fw-login-id input");
              const b = document.querySelector("#fw-signup-id input");
              const uid = (a && a.value) || (b && b.value);
              if (uid) localStorage.setItem("fawry_user_id", uid);
              setTimeout(fwLoadOffers, 100);
            }, 250);
          }
        });
        setInterval(fwLoadOffers, 5000);
"""


def build_app():
    with gr.Blocks(css=CUSTOM_CSS, title="Fawry", theme=gr.themes.Base(), js=FRONTEND_JS) as demo:
        # user_name / logged_in are simple demo-only state
        user_name = gr.State("Ahmed Mohamed")

        with gr.Column(elem_id="fw-shell"):
            login = P.build_login_page()
            signup = P.build_signup_page()

            home = P.build_home_page()
            wallet = P.build_wallet_page()
            offers = P.build_offers_page()
            quickpay = P.build_quickpay_page()

            nav_row, nav_buttons = C.bottom_nav(active="home", visible=False)

        all_page_cols = {
            "login": login["col"], "signup": signup["col"],
            "home": home["col"], "wallet": wallet["col"],
            "offers": offers["col"], "quickpay": quickpay["col"],
        }

        # ---- helpers -------------------------------------------------
        def goto(target):
            """Return a dict of gr.update() for every page column + nav row,
            showing only `target` (nav row only visible for app pages)."""
            # NOTE: keyed by the string page name (was previously keyed by
            # the Column object itself, which caused a KeyError below).
            updates = {key: gr.update(visible=(key == target))
                       for key in all_page_cols}
            updates["nav"] = gr.update(visible=(target in APP_PAGES))
            return updates

        def display_name_from_email(email):
            """No name field on the login form, so derive something readable
            from the email/phone they typed (e.g. 'ahmed.mohamed' -> 'Ahmed Mohamed')."""
            local_part = email.split("@")[0] if "@" in email else email
            cleaned = local_part.replace(".", " ").replace("_", " ").replace("-", " ").strip()
            return cleaned.title() if cleaned else "Fawry User"

        def make_router(target):
            def _router(*_args):
                upd = goto(target)
                return [upd[k] for k in
                        ["login", "signup", "home", "wallet", "offers", "quickpay", "nav"]]
            return _router

        outputs = [login["col"], signup["col"], home["col"],
                   wallet["col"], offers["col"], quickpay["col"], nav_row]

        # ---- auth flow -------------------------------------------------
        # Both handlers also refresh `home["header_html"]` so the Home page
        # greets whoever just logged in / signed up by name, while every
        # other section of Home stays exactly as designed.
        def do_login(email, password):
            ok = bool(email and password)
            msg = "✅ Logged in! Redirecting..." if ok else \
                  "⚠️ Please enter your email/phone and password."
            target = "home" if ok else "login"
            upd = goto(target)
            name = display_name_from_email(email) if ok else "Ahmed Mohamed"
            header_upd = gr.update(value=C.app_header_html(name=name)) if ok else gr.update()
            return [msg, *[upd[k] for k in
                    ["login", "signup", "home", "wallet", "offers", "quickpay", "nav"]],
                    header_upd, name]

        def do_signup(name, phone, email, password):
            ok = all([name, phone, email, password])
            msg = "✅ Account created! Redirecting..." if ok else \
                  "⚠️ Please fill in every field."
            target = "home" if ok else "signup"
            upd = goto(target)
            header_upd = gr.update(value=C.app_header_html(name=name)) if ok else gr.update()
            return [msg, *[upd[k] for k in
                    ["login", "signup", "home", "wallet", "offers", "quickpay", "nav"]],
                    header_upd, name]

        login["login_btn"].click(
            do_login, inputs=[login["email"], login["password"]],
            outputs=[login["status"], *outputs, home["header_html"], user_name],
        )
        signup["signup_btn"].click(
            do_signup,
            inputs=[signup["name"], signup["phone"], signup["email"], signup["password"]],
            outputs=[signup["status"], *outputs, home["header_html"], user_name],
        )
        login["to_signup_btn"].click(make_router("signup"), outputs=outputs)
        signup["to_login_btn"].click(make_router("login"), outputs=outputs)

        # ---- bottom nav routing -----------------------------------------
        nav_buttons["home"].click(make_router("home"), outputs=outputs)
        nav_buttons["wallet"].click(make_router("wallet"), outputs=outputs)
        nav_buttons["offers"].click(make_router("offers"), outputs=outputs)
        nav_buttons["quickpay"].click(make_router("quickpay"), outputs=outputs)
        nav_buttons["more"].click(make_router("wallet"), outputs=outputs)  # demo fallback

        # ---- header shortcut buttons -------------------------------------
        home["top_up_btn"].click(make_router("wallet"), outputs=outputs)
        wallet["add_money_btn"].click(make_router("wallet"), outputs=outputs)

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
