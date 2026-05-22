import requests
from bs4 import BeautifulSoup


class WikipediaFetcher:

    def fetch_page(self, topic):

        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"

        headers = {
            'User-Agent': 'Mozilla/5.0'
        }

        response = requests.get(
            url,
            headers=headers
        )

        if response.status_code != 200:

            print('Failed to fetch Wikipedia page')

            return None

        soup = BeautifulSoup(
            response.text,
            'html.parser'
        )

        title_tag = soup.find('h1')

        title = title_tag.text if title_tag else topic

        # =====================================================
        # SECTION + SUBSECTION EXTRACTION
        # =====================================================

        sections = []

        current_section = "Introduction"

        current_subsection = ""

        for tag in soup.find_all(
            ['h2', 'h3', 'p']
        ):

            # SECTION

            if tag.name == 'h2':

                current_section = tag.get_text(
                    strip=True
                ).replace('[edit]', '')

            # SUBSECTION

            elif tag.name == 'h3':

                current_subsection = tag.get_text(
                    strip=True
                ).replace('[edit]', '')

            # PARAGRAPH

            elif tag.name == 'p':

                paragraph_text = tag.get_text(
                    strip=True
                )

                if paragraph_text:

                    sections.append({

                        "text": paragraph_text,

                        "section": current_section,

                        "subsection": current_subsection

                    })

        # FULL TEXT

        full_text = " ".join([
            item['text']
            for item in sections
        ])

        # =====================================================
        # IMAGE EXTRACTION
        # =====================================================

        images = []

        for img in soup.find_all('img'):

            src = img.get('src')

            if not src:
                continue

            if '.svg' in src:
                continue

            if (
                '.png' not in src
                and '.jpg' not in src
                and '.jpeg' not in src
            ):
                continue

            if src.startswith('//'):
                src = 'https:' + src

            images.append(src)

        print(f"Found {len(images)} images")

        # =====================================================
        # AUDIO EXTRACTION
        # =====================================================

        audio_files = []

        for audio in soup.find_all('audio'):

            source = audio.find('source')

            if source and source.get('src'):

                src = source.get('src')

                if src.startswith('//'):
                    src = 'https:' + src

                audio_files.append(src)

        print(f"Found {len(audio_files)} audio files")

        # =====================================================
        # VIDEO EXTRACTION
        # =====================================================

        videos = []

        for video in soup.find_all('video'):

            source = video.find('source')

            if source and source.get('src'):

                src = source.get('src')

                if src.startswith('//'):
                    src = 'https:' + src

                videos.append(src)

        print(f"Found {len(videos)} video files")

        return {

            "title": title,

            "text": full_text,

            "sections": sections,

            "source": url,

            "images": images[:10],

            "audio": audio_files[:10],

            "videos": videos[:10]
        }


fetcher = WikipediaFetcher()