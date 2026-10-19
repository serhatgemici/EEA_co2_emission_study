import streamlit as st

profiles = [
    {
        "name": "Serhat Gemici",
        "avatar": "https://avatars.githubusercontent.com/u/19411404?v=4",
        "linkedin": "https://www.linkedin.com/in/serhatgemici/",
        "github": "https://github.com/serhatgemici/",
        "about": (
            "Former Senior QA Lead with 12+ years of experience in software quality engineering, test automation, and CI/CD. "
            "\n\n"
            "Building on a strong background in quality engineering, automation, and software delivery, I am moving into ML & AI Engineering while completing Liora's Machine Learning Engineer program with Université Paris-Sorbonne."
            "\n\n"
            "Developing hands-on expertise in Python-based data analysis, statistical modeling, machine learning, deep learning, and MLOps, including the engineering practices required to deploy and monitor ML solutions. "
            "\n\n"
            "Let's connect and collaborate on exciting ML & AI projects!"
        ),
    },
    {
        "name": "Ana M. Valencia",
        "avatar": "https://avatars.githubusercontent.com/u/312455389?v=4",
        "linkedin": "https://www.linkedin.com/in/ana-m-valencia/",
        "github": "https://github.com/anamvg/",
        "about": (
            "Guest scientist and independent researcher at the University of Oldenburg, specializing in theoretical and computational condensed matter physics. "
            "\n\n"
            "My research focuses on the multiscale modelling of hybrid interfaces and functional materials using state-of-the-art DFT and MBPT. "
            "\n\n"
            "With extensive project and leadership experience, I have successfully formulated independent research proposals and secured substantial high-performance computing resources, including nearly 9.3 million core-hours from the NHR Center. "
            "\n\n"
            "Currently, I am expanding my technical expertise by completing a Data Science certification through LIORA to bridge high-performance simulation with data-driven methodologies."
        ),
    },
]


def render_profile(profile):
    about_html = profile["about"].replace(
        "\n\n",
        "</p><p>",
    )
    st.html(f"""
        <div class="profile">
            <a href="{profile['linkedin']}" target="_blank">
                <img
                    src="{profile['avatar']}"
                    alt="{profile['name']}"
                    class="avatar"
                />
            </a>

            <h3>{profile['name']}</h3>

            <p class="linkedin-label">
                <a href="{profile['linkedin']}" target="_blank">
                    LinkedIn
                </a>
            </p>
            <p class="linkedin-label">
                <a href="{profile['github']}" target="_blank">
                    GitHub
                </a>
            </p>

            <div class="about">
                <p>{about_html}</p>
            </div>
        </div>

        <style>
            .profile {{
                text-align: center;
                padding: 1rem;
            }}

            .avatar {{
                width: 128px;
                height: 128px;
                border-radius: 50%;
                object-fit: cover;
                display: block;
                margin: 0 auto 1rem auto;
            }}

            .profile h3 {{
                margin: 0.5rem 0;
            }}

            .linkedin-label {{
                color: #666;
                margin: 0.25rem 0 1rem 0;
            }}

                .linkedin-label a,
                .linkedin-label a:link,
                .linkedin-label a:visited,
                .linkedin-label a:hover,
                .linkedin-label a:active,
                .linkedin-label a:focus {{
                    color: #5F6875;
                    text-decoration: underline;
                }}

            .about {{
                line-height: 1.6;
                margin: 1rem auto 0 auto;
                max-width: 600px;
                padding: 0 1rem;
                text-align: left;
            }}
        </style>
        """)


col1, separator, col2 = st.columns(
    [2, 0.1, 2],
    gap="large",
    vertical_alignment="top",
)

with col1:
    render_profile(profiles[0])

with separator:
    st.html("""
        <div
            style="
                border-left: 1px solid rgba(49, 51, 63, 0.2);
                height: 360px;
                margin: 160px auto 0 auto;
            "
        ></div>
        """)

with col2:
    render_profile(profiles[1])
