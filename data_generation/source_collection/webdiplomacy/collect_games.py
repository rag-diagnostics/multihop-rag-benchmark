import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os
import csv
import time
from urllib.parse import urljoin
from pathlib import Path
import argparse

class WebDiplomacyScraper:
    def __init__(self, output_dir="work", base_url="https://webdiplomacy.net"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.messages_dir = self.output_dir / "game_messages"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create directories
        self.messages_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV file
        self.csv_file = str(self.output_dir / 'game_metadata.csv')
        self.init_csv()
    
    def init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'game_id', 'finish_date', 'countries_mapping', 'total_messages', 
                'conversation_pairs', 'message_pages_scraped', 'message_file_path'
            ])
    
    def parse_date(self, date_text):
        """Parse date from various formats and return year"""
        try:
            # Try different date formats
            formats = [
                "%a, %I:%M %p, %d %b %Y",  # Wed, 11:13 AM, 4 Jan 2023
                "%d %b %Y",                # 4 Jan 2023
                "%Y-%m-%d"                 # 2023-01-04
            ]
            
            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_text.strip(), fmt)
                    return parsed_date.year, parsed_date.strftime("%Y-%m-%d")
                except ValueError:
                    continue
            
            # Extract year using regex if formats fail
            year_match = re.search(r'\b(20\d{2})\b', date_text)
            if year_match:
                year = int(year_match.group(1))
                return year, f"{year}-01-01"
            
        except Exception as e:
            print(f"Error parsing date '{date_text}': {e}")
        
        return None, None
    
    def extract_game_info(self, game_panel):
        """Extract game ID, finish date, and country mappings from game panel"""
        try:
            # Extract game ID from board.php link
            game_id = None
            view_link = game_panel.find('a', href=re.compile(r'board\.php\?gameID=\d+'))
            if view_link:
                game_id_match = re.search(r'gameID=(\d+)', view_link['href'])
                if game_id_match:
                    game_id = int(game_id_match.group(1))
            
            if not game_id:
                return None
            
            # Extract finish date
            finish_date = None
            finish_year = None
            timestamp = game_panel.find('span', class_='timestampGames')
            if timestamp and timestamp.has_attr('unixtime'):
                unix_time = int(timestamp['unixtime'])
                dt = datetime.fromtimestamp(unix_time)
                finish_year = dt.year
                finish_date = dt.strftime("%Y-%m-%d")
            else:
                # Fallback to text parsing
                date_spans = game_panel.find_all('span', class_='timestampGames')
                for span in date_spans:
                    if span.text:
                        finish_year, finish_date = self.parse_date(span.text)
                        break
            
            # Only process games from 2025
            if not finish_year or finish_year != 2025:
                return None
            
            # Extract country mappings
            country_mapping = {}
            member_rows = game_panel.find_all('tr', class_='member')
            
            for row in member_rows:
                country_span = row.find('span', class_=re.compile(r'country\d+'))
                if country_span:
                    # Extract country number from class
                    country_class = [cls for cls in country_span['class'] if cls.startswith('country')][0]
                    country_number = country_class.replace('country', '')
                    country_name = country_span.get_text().strip()
                    
                    if country_name and country_number:
                        country_mapping[country_number] = country_name
            
            return {
                'game_id': game_id,
                'finish_date': finish_date,
                'country_mapping': country_mapping
            }
            
        except Exception as e:
            print(f"Error extracting game info: {e}")
            return None
    
    def get_messages_from_page(self, game_id, page_num, country_mapping):
        """Get messages from a specific page"""
        messages_url = f"{self.base_url}/board.php?gameID={game_id}&viewArchive=Messages&pagenum={page_num}#results"
        
        try:
            response = self.session.get(messages_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find message table
            message_table = soup.find('table', class_='archive-messages-table')
            if not message_table:
                return []
            
            messages = []
            rows = message_table.find_all('tr')
            
            for row in rows:
                try:
                    # Extract timestamp
                    timestamp_span = row.find('span', class_='timestamp')
                    date_text = ""
                    unix_timestamp = None
                    
                    if timestamp_span:
                        if timestamp_span.has_attr('unixtime'):
                            unix_timestamp = int(timestamp_span['unixtime'])
                            dt = datetime.fromtimestamp(unix_timestamp)
                            date_text = dt.strftime("%d %b %Y")
                        else:
                            date_text = timestamp_span.get_text().strip()
                    
                    # Extract country and message
                    country_td = row.find('td', class_=re.compile(r'country\d+'))
                    if country_td:
                        # Extract country number from class
                        country_classes = [cls for cls in country_td['class'] if cls.startswith('country')]
                        if country_classes:
                            country_number = country_classes[0].replace('country', '')
                            country_name = country_mapping.get(country_number, f'Country{country_number}')
                            message_text = country_td.get_text().strip()
                            
                            # Clean up message text
                            message_text = re.sub(r'\s+', ' ', message_text).strip()
                            
                            if message_text:  # Only add non-empty messages
                                # Try to extract season and year from various sources
                                season = None
                                year = None
                                
                                # Check row class for season/year info
                                row_class = ' '.join(row.get('class', []))
                                season_match = re.search(r'season(\w+)', row_class)
                                year_match = re.search(r'year(\d+)', row_class)
                                
                                if season_match:
                                    season = season_match.group(1)
                                if year_match:
                                    year = year_match.group(1)
                                
                                # Try to extract from gameID class
                                gameid_match = re.search(r'gameID(\d+)countryID(\d+)', row_class)
                                if gameid_match:
                                    extracted_game_id = gameid_match.group(1)
                                    extracted_country_id = gameid_match.group(2)
                                    # Verify this matches our expected game and country
                                    if extracted_game_id == str(game_id) and extracted_country_id == country_number:
                                        pass  # Validation passed
                                
                                messages.append({
                                    'date': date_text,
                                    'country': country_name,
                                    'country_code': country_number,
                                    'message': message_text,
                                    'season': season,
                                    'year': year,
                                    'timestamp': unix_timestamp,
                                    'page': page_num
                                })
                                
                except Exception as e:
                    print(f"Error processing row in game {game_id}, page {page_num}: {e}")
                    continue
            
            return messages
            
        except Exception as e:
            print(f"Error getting messages for game {game_id}, page {page_num}: {e}")
            return []
    
    def get_all_game_messages(self, game_id, country_mapping, max_pages=15):
        """Get all messages for a specific game across multiple pages"""
        all_messages = []
        pages_scraped = 0
        
        for page_num in range(1, max_pages + 1):
            print(f"  Scraping messages page {page_num} for game {game_id}...")
            
            messages = self.get_messages_from_page(game_id, page_num, country_mapping)
            
            if not messages:
                print(f"  No messages found on page {page_num}, stopping pagination")
                break
            
            all_messages.extend(messages)
            pages_scraped = page_num
            
            print(f"  Found {len(messages)} messages on page {page_num}")
            
            # Add small delay between page requests
            time.sleep(0.5)
        
        # Sort messages by timestamp if available
        def sort_key(msg):
            if msg['timestamp']:
                return msg['timestamp']
            return 0
        
        all_messages.sort(key=sort_key)
        
        return all_messages, pages_scraped
    
    def format_messages_by_conversation(self, messages, country_mapping):
        """Format messages grouped by country pairs"""
        conversations = {}
        
        # Create all possible country pairs
        countries = list(country_mapping.values())
        for i, country1 in enumerate(countries):
            for j, country2 in enumerate(countries):
                if i < j:  # Avoid duplicates
                    pair = tuple(sorted([country1.lower(), country2.lower()]))
                    conversations[pair] = []
        
        # Add messages to conversations
        for msg in messages:
            country = msg['country'].lower()
            for pair in conversations.keys():
                if country in pair:
                    conversations[pair].append(msg)
        
        # Remove empty conversations and sort messages within each conversation
        final_conversations = {}
        for pair, msgs in conversations.items():
            if msgs:
                # Sort by timestamp
                msgs.sort(key=lambda x: x['timestamp'] if x['timestamp'] else 0)
                final_conversations[pair] = msgs
        
        return final_conversations
    
    def save_game_messages(self, game_id, conversations):
        """Save messages to text file"""
        filename = str(self.messages_dir / f'game_{game_id}_messages.txt')
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for pair, messages in conversations.items():
                    if not messages:
                        continue
                    
                    f.write(f"--- Game ID: {game_id} | Players: {', '.join(pair)} ---\n\n")
                    
                    # Group messages by season/year if available
                    current_season = None
                    current_year = None
                    message_count = 0
                    
                    for msg in messages:
                        # Check if we have season/year info
                        if msg.get('season') and msg.get('year'):
                            season_year = f"{msg['season']}|{msg['year']}"
                            if season_year != f"{current_season}|{current_year}":
                                current_season = msg['season']
                                current_year = msg['year']
                                f.write(f"\n  Season: {current_season} | Year: {current_year}\n")
                        elif current_season is None:
                            # If no season info, group by date
                            if msg['date']:
                                f.write(f"\n  Date: {msg['date']}\n")
                        
                        f.write(f"{msg['country']}: {msg['message']}\n")
                        message_count += 1
                    
                    f.write(f"\n[Total messages in this conversation: {message_count}]\n")
                    f.write("\n" + "="*80 + "\n\n")
            
            return filename
            
        except Exception as e:
            print(f"Error saving messages for game {game_id}: {e}")
            return None
    
    def save_metadata(self, game_info, total_messages, conversation_pairs, pages_scraped, file_path):
        """Save metadata to CSV"""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    game_info['game_id'],
                    game_info['finish_date'],
                    str(game_info['country_mapping']),
                    total_messages,
                    conversation_pairs,
                    pages_scraped,
                    file_path
                ])
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def scrape_finished_games(self, max_pages=10, max_message_pages=15):
        """Scrape finished games from multiple pages"""
        games_processed = 0
        total_messages_processed = 0
        
        for page_num in range(1, max_pages + 1):
            print(f"\nProcessing game listings page {page_num}...")
            
            url = f"{self.base_url}/gamelistings.php?sortCol=turn&sortType=desc&gamelistType=Finished&pagenum={page_num}#results"
            
            try:
                response = self.session.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all game panels
                game_panels = soup.find_all('div', class_='gamePanel')
                
                if not game_panels:
                    print(f"No game panels found on page {page_num}")
                    break
                
                page_games_found = 0
                for panel in game_panels:
                    game_info = self.extract_game_info(panel)
                    
                    if game_info:  # Only 2025 games
                        page_games_found += 1
                        print(f"\nProcessing game {game_info['game_id']} (2025 game)...")
                        print(f"Countries: {game_info['country_mapping']}")
                        
                        # Get all messages across multiple pages
                        messages, pages_scraped = self.get_all_game_messages(
                            game_info['game_id'], 
                            game_info['country_mapping'], 
                            max_message_pages
                        )
                        
                        if messages:
                            # Format conversations
                            conversations = self.format_messages_by_conversation(
                                messages, 
                                game_info['country_mapping']
                            )
                            
                            # Save messages
                            file_path = self.save_game_messages(game_info['game_id'], conversations)
                            
                            # Save metadata
                            total_messages = len(messages)
                            conversation_pairs = len(conversations)
                            
                            self.save_metadata(
                                game_info, 
                                total_messages, 
                                conversation_pairs, 
                                pages_scraped, 
                                file_path
                            )
                            
                            games_processed += 1
                            total_messages_processed += total_messages
                            print(f"✓ Saved game {game_info['game_id']}: {total_messages} messages across {pages_scraped} pages")
                        else:
                            print(f"✗ No messages found for game {game_info['game_id']}")
                        
                        # Add delay between games to be respectful
                        time.sleep(1)
                
                print(f"Found {page_games_found} games from 2025 on this page")
                
                # If no 2025 games found on this page, we might be getting into older games
                if page_games_found == 0:
                    print("No 2025 games found on this page, likely reached older games")
                    
                # Add delay between pages
                time.sleep(2)
                
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")
                continue
        
        print(f"\n" + "="*60)
        print(f"SCRAPING COMPLETED!")
        print(f"Total games processed: {games_processed}")
        print(f"Total messages processed: {total_messages_processed}")
        print(f"Average messages per game: {total_messages_processed/games_processed if games_processed > 0 else 0:.1f}")

def main():
    parser = argparse.ArgumentParser(description="Historical 2025 WebDiplomacy message collector")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write game_messages/ and game_metadata.csv")
    parser.add_argument("--max-listing-pages", type=int, default=1,
                        help="Maximum finished-game listing pages (default 1; historical notebook used 5000)")
    parser.add_argument("--max-message-pages", type=int, default=15,
                        help="Maximum archive pages per game (default 15)")
    args = parser.parse_args()
    if args.max_listing_pages < 1 or args.max_message_pages < 1:
        parser.error("Page limits must be positive")

    scraper = WebDiplomacyScraper(output_dir=args.output_dir)
    scraper.scrape_finished_games(max_pages=args.max_listing_pages,
                                 max_message_pages=args.max_message_pages)
    print("\nCollection finished; see the requested output directory for files and metadata")


if __name__ == "__main__":
    main()
