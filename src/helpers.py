

import re
from bs4 import BeautifulSoup

data = pd.read_json('data/ufo_first100records.json', lines=True)
soup = BeautifulSoup(data.html[0], "html.parser")
table = soup.find("tbody")
table.text.strip()

"Occurred : 5/6/2017 05:00  (Entered as : 05/06/2017 05:00)Reported: 5/6/2017 4:10:01 AM 04:10Posted: 5/6/2017Location: Camp McGregor, NMShape: LightDuration:10 minutes\n\n\nLight seen over mountain's east of Camp McGregor.  It hovered in one spot.  It looked like a helicopter light at first.  But it was way to bright.  Then it went to a higher height.  Stayed there for a while.  Then disappeared.((NUFORC Note:  Report appears to us to be consistent with the sighting of Venus.  PD))"

'Occurred : 5/6/2017 04:50  (Entered as : 06/05/2017 04:50)Reported: 5/6/2017 5:00:54 AM 05:00Posted: 5/6/2017Location: Mojave (Canada), BCShape: LightDuration:1:00\n\n\nLight in sky stationary.  Not a airplane or any known star.  Flickering in brightness.((NUFORC Note:  Possibly Venus.  Source of the report elects to remain anonymous; provides no contact information.  We could not find a city named "Mojave" in BC, Canada.  PD))'

"Occurred : 4/30/2017 21:00  (Entered as : 04/30/17 21:00)Reported: 5/1/2017 4:58:11 AM 04:58Posted: 5/6/2017Location: Ventura, CAShape: OtherDuration:\n\n\nThere was one bigger light flashing with about 13 smaller ones except the smaller ones flashed a bit different.  It reminded me of a drone, but over a dozen drones out flying above our neighborhood all night until the morning is a bit odd.  I went to bed at about 23:00 and woke up at 4:00, they were still out there.  One flew over my fiancés and my head and we saw a shape of  what appeared to be a triangle shape.  My dad and mom both saw them from their window and they thought it was quite strange.  I can't believe no one has reported this yet.  I don't know what this is, but they are circling our neighborhood over and over again.  Hovering and they haven't stopped."
def data_pipeline(data):
    '''
    input: raw dataframe with html column

    '''

    contents = []
    for row in data.html:
        soup = BeautifulSoup(row, "html.parser")
        table = soup.find("tbody")
        text = table.text.strip()
        a = re.findall(r'(Duration.+?\n\n\n)(.+)(\(\(NUFORC Note:)?', text)
        try:
            contents.append(a[0][1])
        except:
            pass

    return contents
