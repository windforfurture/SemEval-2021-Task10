import os

from bs4 import BeautifulSoup
import requests
import json
from datetime import datetime
import json
import re

from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

print(tokenizer(["next","before"], is_split_into_words=True,return_offsets_mapping=True, padding=True, truncation=True)['input_ids'])

# url_list = ['/2020/11/27/politics/supreme-court-justices-personal-tone/index.html', '/2020/11/26/politics/andrew-cuomo-scotus-coronavirus/index.html', '/2020/11/26/politics/supreme-court-religious-restrictions-ruling-covid/index.html', '/2020/11/26/politics/supreme-court-ruling-on-religious-dispute-in-ny/index.html', '/2020/11/27/politics/trump-biden-coronavirus-economy/index.html', '/2020/11/27/world/trump-election-defeat-populism-global-intl/index.html', '/2020/11/26/asia/hong-kong-eggs-law-intl-hnk/index.html', '/2020/11/27/business/china-australia-wine-dumping-intl-hnk/index.html', '/2020/11/27/asia/china-university-thanksgiving-intl-hnk/index.html', '/2020/11/27/africa/ghana-president-daughter-social-media-intl/index.html', '/2020/11/27/business/switzerland-responsible-business-initiative-referendum/index.html', '/2020/11/26/europe/france-police-violence-video-intl/index.html', '/2020/11/26/football/diego-maradona-death-spt-intl/index.html', '/2020/11/27/sport/markus-paul-tributes-dallas-cowboys-nfl-spt-intl/index.html', '/2020/11/26/europe/turkey-attempted-coup-trial-intl/index.html', '/2020/11/27/football/harry-winks-goal-tottenham-hotspur-ludogorets-europa-league-spt-intl/index.html', '/2020/11/27/investing/premarket-stocks-trading/index.html', '/2020/11/27/health/us-coronavirus-friday/index.html', '/2020/11/26/health/states-coronavirus-exposure-notifications-trnd/index.html', '/2020/11/27/health/tiktok-therapist-mental-health-intl-wellness/index.html', '/2020/11/26/us/national-dog-show-2020-trnd/index.html', '/2020/11/26/us/alex-trebek-thanksgiving-message/index.html', '/2020/11/27/business/food-climate-label-carbon-footprint-spc-intl/index.html', '/2020/11/26/football/diego-maradona-death-argentina-spt-intl/index.html', '/2020/10/16/world/gallery/science-fiction-inventions-became-reality-spc-intl/index.html', '/2020/11/27/world/neanderthal-human-hands-thumb-grip-scn/index.html', '/2020/11/27/health/thanksgiving-leftovers-ideas-wellness/index.html', '/2020/11/03/world/gallery/robots-helping-people-tech-for-good-spc-intl/index.html', '/2020/09/17/world/captive-breeding-species-cte-scn-spc-intl/index.html']
# url_list = ['/2020/11/28/asia/japan-suicide-women-covid-dst-intl-hnk/index.html', '/2020/11/28/health/oxford-astrazeneca-vaccines-developing-countries-intl/index.html', '/2020/11/29/politics/coronavirus-vaccine-distribution-congress-funding/index.html', '/2020/11/29/sport/nfl-denver-broncos-quarterbacks-ineligible/index.html', '/2020/11/29/politics/biden-economic-policies-covid/index.html', '/2020/11/28/politics/pennsylvania-state-supreme-court-election-case/index.html', '/2020/11/28/us/utah-monolith-disappears-trnd/index.html', '/2020/11/28/middleeast/iran-mohsen-fakhrizadeh-killing-analysis-intl/index.html', '/2020/11/28/europe/france-protests-security-law-intl/index.html', '/2020/11/28/africa/ethiopia-tigray-bombardment-intl/index.html', '/2020/11/29/europe/chopin-sexuality-poland-lgbtq-debate-scli-intl/index.html', '/2020/11/28/us/rowdy-harrell-blakley-killed-crash-hendrick-motorsports-trnd/index.html', '/2020/11/29/sport/mike-tyson-roy-jones-jr-wins-frontline-battle-trnd/index.html', '/2020/11/29/entertainment/david-prowse-darth-vadar-star-wars-dies-gbr-scli-intl/index.html', '/2020/11/28/us/john-travolta-thanksgiving-message-trnd/index.html', '/2020/11/29/entertainment/dance-monkey-shazam-record-trnd/index.html', '/2020/11/27/business/lego-colosseum-trnd/index.html', '/2020/11/29/australia/australia-fire-danger-intl-hnk/index.html', '/2020/11/29/world/lunar-eclipse-full-beaver-moon-2020-scn-trnd/index.html', '/2020/11/28/us/black-santa-decoration-racist-letter-trnd/index.html', '/2020/11/27/health/cancer-blood-test-pilot-gbr-intl/index.html', '/2020/11/29/sport/dan-carter-mental-health-rugby-spt-intl-cmd/index.html', '/2020/11/27/asia/north-korea-astrazeneca-suspected-cyberattack-intl/index.html', '/2020/11/28/football/barcelona-real-madrid-el-clasico-2010-anniversary-cmd-spt-intl/index.html', '/2020/11/25/tech/tech-gadgets-2020-cnn-staff-picks/index.html', '/2020/11/27/business/food-climate-label-carbon-footprint-spc-intl/index.html', '/2020/11/27/app-news-section/videos-of-the-week-mobile-nov-27/index.html', '/2020/11/28/tech/victor-glover-space-x-astronaut-video-earth-scn-trnd/index.html', '/2020/11/29/entertainment/christmas-movies-2020-trnd/index.html', '/2020/11/27/health/tiktok-therapist-mental-health-intl-wellness/index.html', '/2020/11/03/world/gallery/robots-helping-people-tech-for-good-spc-intl/index.html', '/2020/09/17/world/captive-breeding-species-cte-scn-spc-intl/index.html']
# url_list = ['/2020/11/30/politics/donald-trump-joe-biden-coronavirus-economy/index.html', '/2020/11/30/health/moderna-vaccine-fda-eua-application/index.html', '/2020/11/30/health/us-coronavirus-monday/index.html', '/2020/11/29/middleeast/iran-mohsen-fakhrizadeh-remote-control-machine-gun/index.html', '/2020/11/30/uk/coronavirus-england-lockdown-uk-gbr-intl/index.html', '/2020/11/30/media/trump-election-confusion-reliable-sources/index.html', '/2020/11/30/politics/census-supreme-court-oral-arguments/index.html', '/2020/11/30/australia/australia-china-twitter-intl-hnk/index.html', '/2020/11/30/asia/sri-lanka-covid-19-prison-intl/index.html', '/2020/11/29/politics/biden-twisted-ankle/index.html', '/2020/11/30/football/diego-maradona-death-doctor-investigation-spt-intl/index.html', '/2020/11/30/business/hong-kong-carrie-lam-cash-intl-hnk/index.html', '/2020/11/30/entertainment/the-crown-fiction-warning-scli-gbr-intl/index.html', '/2020/11/30/asia/thailand-protest-lese-majeste-intl-hnk/index.html', '/2020/11/30/asia/stuff-maori-apology-intl-scli/index.html', '/2020/11/30/europe/uk-woman-missing-pyrenees-scli-intl-gbr/index.html', '/2020/11/30/football/papa-bouba-diop-senegal-world-cup-death-spt-intl/index.html', '/2020/11/30/business/australia-china-wine-tariffs-intl-hnk/index.html', '/2020/11/30/asia/white-island-volcano-charges-intl-hnk/index.html', '/2020/11/30/us/coronavirus-indigenous-farmworkers-radio-station/index.html', '/2020/11/30/health/mental-health-black-community-wellness-partner/index.html', '/2020/11/30/world/coronavirus-newsletter-11-30-20-intl/index.html', '/2020/11/30/health/dealing-with-grief-pandemic-wellness/index.html', '/2020/11/30/sport/tom-brady-super-bowl-patrick-mahomes-nfl-spt-intl/index.html', '/2020/11/29/asia/pranav-lal-blind-photographer-spc-intl/index.html', '/2020/11/27/europe/fall-leaves-intl-scli-climate-scn/index.html', '/2020/11/30/motorsport/romain-grosjean-crash-halo-bahrain-gp-f1-spt-intl/index.html', '/2020/11/24/business/africa-e-logistics-lori-systems-spc-intl/index.html', '/2020/11/28/tech/victor-glover-space-x-astronaut-video-earth-scn-trnd/index.html', '/2020/11/03/world/gallery/robots-helping-people-tech-for-good-spc-intl/index.html', '/2020/09/17/world/captive-breeding-species-cte-scn-spc-intl/index.html']


# site_url = 'https://edition.cnn.com/'
# web_data = requests.get(site_url)
# web_data.encoding = 'utf-8'
# soup = BeautifulSoup(web_data.text, 'lxml')
# # print(soup)
# content = str(soup.find_all('script')[2])
# key_name = "articleList"
# begin_idx = content.index(key_name)
# end_idx = content[begin_idx:].index("]")
# json_data = json.loads(content[begin_idx+len(key_name)+2:end_idx+begin_idx+1])
# url_list = []
# for item in json_data:
#     if item['uri'].startswith('/2020'):
#         url_list.append(item['uri'])
# print(url_list)

# for url_i in url_list:
#     site_url = 'https://edition.cnn.com' + url_i
#     web_data = requests.get(site_url)
#     web_data.encoding = 'utf-8'
#     soup = BeautifulSoup(web_data.text, 'lxml')
#     soup_article = soup.select('.pg-right-rail-tall')
#     if len(soup_article) == 0:
#         continue
#     main_content = soup_article[0]
#     content = main_content.select('article')[0].select('meta')
#     paragraph = main_content.select('.zn-body__paragraph')
#     paper_content = ""
#     for one_para in paragraph:
#         paper_content += one_para.text + '\n' + '\r\n'
#
#     title = ""
#     author = ""
#     datePublished = ""
#     for one_content in content:
#         content_dict = one_content.attrs
#         itemprop = content_dict.get('itemprop')
#         if content_dict.get('itemprop') == 'headline':
#             title = content_dict.get("content")
#         if content_dict.get('itemprop') == 'author':
#             author = content_dict.get("content")
#         if content_dict.get('itemprop') == 'datePublished':
#             datePublished = content_dict.get("content")
#     content_time = datePublished[:4]+datePublished[5:7]+datePublished[8:10]
#     path_no = datePublished[11:13]+datePublished[14:16]+datePublished[17:19]
#     print(title, author, datePublished)
#     file_name = "CNN_"+content_time+"_"+path_no
#     file_path = os.path.join("new_train_data", "time", "CNN", file_name)
#     if not os.path.exists(file_path):
#         os.makedirs(file_path)
#         print(file_path)
#         with open(os.path.join(file_path, file_name), "w") as output_file:
#             output_file.write('\n')
#             output_file.write('\n')
#             output_file.write(file_name)
#             output_file.write('\n')
#             output_file.write('\n')
#             output_file.write(datePublished)
#             output_file.write('\n')
#             output_file.write('\n')
#             output_file.write(title)
#             output_file.write('\n')
#             output_file.write('\n')
#             output_file.write(paper_content)