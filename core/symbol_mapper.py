# core/symbol_mapper.py
# Author: Rafał Wiśniewski | Data & AI Solutions

import re
from typing import List, Dict, Optional, Set
import logging

class SymbolMapper:
    """
    Mapowanie symboli z prefiksami/sufiksami brokerów
    np. EURUSD.a, EURUSDm, #EURUSD, EURUSD_i, etc.
    """
    
    # Znane pary bazowe
    MAJOR_PAIRS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'EURAUD', 'EURCHF', 'AUDNZD',
        'NZDJPY', 'GBPAUD', 'GBPCAD', 'EURNZD', 'AUDCAD', 'GBPCHF', 'AUDCHF',
        'EURCAD', 'CADJPY', 'GBPNZD', 'CADCHF', 'CHFJPY', 'NZDCAD', 'NZDCHF'
    ]
    
    # Znane indeksy/metale/commodities
    OTHER_INSTRUMENTS = [
        'XAUUSD', 'XAGUSD', 'GOLD', 'SILVER',  # Metale
        'BTCUSD', 'ETHUSD', 'LTCUSD',  # Crypto
        'US30', 'US500', 'NAS100', 'GER40', 'UK100', 'JPN225',  # Indeksy
        'USOIL', 'UKOIL', 'BRENT', 'WTI',  # Ropa
    ]
    
    ALL_BASE_SYMBOLS = MAJOR_PAIRS + OTHER_INSTRUMENTS
    
    # Znane prefiksy/sufiksy brokerów
    KNOWN_PREFIXES = [
        '#', '_', '.', '-',
        'f', 'm', 'i', 'c', 'pro', 'ecn', 'stp', 'raw', 'prime',
        'mini', 'micro', 'std', 'cent'
    ]
    
    KNOWN_SUFFIXES = [
        '.', '-', '_',
        'a', 'b', 'c', 'd', 'e', 'f', 'm', 'i', 'pro', 'ecn', 'raw',
        'mini', 'micro', 'cent', 'sb', 'lmax', 'fxcm', 'icm'
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._broker_cache: Dict[str, str] = {}  # {broker_symbol: base_symbol}
        self._detected_pattern: Optional[str] = None
    
    def normalize_symbol(self, broker_symbol: str) -> str:
        """
        Konwertuj symbol brokera na standardowy
        
        Przykłady:
        - EURUSD.a -> EURUSD
        - EURUSDm -> EURUSD
        - #EURUSD -> EURUSD
        - EURUSD_i -> EURUSD
        - EURUSDpro -> EURUSD
        - Gold -> XAUUSD
        """
        broker_symbol = broker_symbol.strip().upper()
        
        # Check cache
        if broker_symbol in self._broker_cache:
            return self._broker_cache[broker_symbol]
        
        # Already standard
        if broker_symbol in self.ALL_BASE_SYMBOLS:
            return broker_symbol
        
        # Try to extract base symbol
        base_symbol = self._extract_base_symbol(broker_symbol)
        
        if base_symbol:
            self._broker_cache[broker_symbol] = base_symbol
            return base_symbol
        
        # Fallback: return as-is
        self.logger.warning(f"⚠️ Could not normalize: {broker_symbol}")
        return broker_symbol
    
    def _extract_base_symbol(self, broker_symbol: str) -> Optional[str]:
        """Ekstraktuj bazowy symbol"""
        
        # Method 1: Remove known prefixes
        for prefix in self.KNOWN_PREFIXES:
            if broker_symbol.startswith(prefix):
                candidate = broker_symbol[len(prefix):]
                if candidate in self.ALL_BASE_SYMBOLS:
                    return candidate
        
        # Method 2: Remove known suffixes
        for suffix in self.KNOWN_SUFFIXES:
            if broker_symbol.endswith(suffix):
                candidate = broker_symbol[:-len(suffix)]
                if candidate in self.ALL_BASE_SYMBOLS:
                    return candidate
        
        # Method 3: Find base symbol within string
        for base in self.ALL_BASE_SYMBOLS:
            if base in broker_symbol:
                return base
        
        # Method 4: Special cases
        special_map = {
            'GOLD': 'XAUUSD',
            'SILVER': 'XAGUSD',
            'BRENT': 'UKOIL',
            'WTI': 'USOIL',
            'BITCOIN': 'BTCUSD',
            'ETHEREUM': 'ETHUSD'
        }
        
        for key, value in special_map.items():
            if key in broker_symbol:
                return value
        
        # Method 5: Regex pattern matching (6-char pairs)
        # Pattern: 3 currency letters + 3 currency letters
        match = re.search(r'([A-Z]{3})([A-Z]{3})', broker_symbol)
        if match:
            candidate = match.group(1) + match.group(2)
            if candidate in self.ALL_BASE_SYMBOLS:
                return candidate
        
        return None
    
    def detect_broker_pattern(self, symbols: List[str]) -> Optional[str]:
        """
        Wykryj wzorzec prefiksu/sufiksu dla wszystkich symboli
        
        Returns:
        - "prefix_#" jeśli wszystkie mają prefix #
        - "suffix_.a" jeśli wszystkie mają sufiks .a
        - None jeśli brak wspólnego wzorca
        """
        if not symbols:
            return None
        
        # Check for common prefix
        common_prefix = None
        for prefix in self.KNOWN_PREFIXES:
            if all(s.startswith(prefix) for s in symbols):
                common_prefix = f"prefix_{prefix}"
                break
        
        # Check for common suffix
        common_suffix = None
        for suffix in self.KNOWN_SUFFIXES:
            if all(s.endswith(suffix) for s in symbols):
                common_suffix = f"suffix_{suffix}"
                break
        
        pattern = common_prefix or common_suffix
        
        if pattern:
            self._detected_pattern = pattern
            self.logger.info(f"🔍 Detected broker pattern: {pattern}")
        
        return pattern
    
    def batch_normalize(self, broker_symbols: List[str]) -> Dict[str, str]:
        """
        Normalizuj wiele symboli jednocześnie
        
        Returns:
        {
            'EURUSD.a': 'EURUSD',
            'GBPUSDm': 'GBPUSD',
            ...
        }
        """
        # Auto-detect pattern
        self.detect_broker_pattern(broker_symbols)
        
        mapping = {}
        for symbol in broker_symbols:
            normalized = self.normalize_symbol(symbol)
            mapping[symbol] = normalized
        
        self.logger.info(f"✅ Normalized {len(mapping)} symbols")
        
        return mapping
    
    def reverse_normalize(self, base_symbol: str, broker_symbols: List[str]) -> Optional[str]:
        """
        Znajdź symbol brokera dla danego bazowego symbolu
        
        Args:
            base_symbol: 'EURUSD'
            broker_symbols: ['EURUSD.a', 'GBPUSD.a', ...]
        
        Returns:
            'EURUSD.a'
        """
        # Build reverse map
        reverse_map = self.batch_normalize(broker_symbols)
        
        for broker_sym, base_sym in reverse_map.items():
            if base_sym == base_symbol:
                return broker_sym
        
        return None
    
    def get_all_variants(self, base_symbol: str) -> List[str]:
        """
        Generuj wszystkie możliwe warianty symbolu z prefiksami/sufiksami
        
        Args:
            base_symbol: 'EURUSD'
        
        Returns:
            ['EURUSD', 'EURUSD.a', 'EURUSDm', '#EURUSD', ...]
        """
        variants = [base_symbol]
        
        # Add prefixes
        for prefix in self.KNOWN_PREFIXES:
            variants.append(f"{prefix}{base_symbol}")
        
        # Add suffixes
        for suffix in self.KNOWN_SUFFIXES:
            variants.append(f"{base_symbol}{suffix}")
        
        return variants
    
    def find_symbol_in_list(self, base_symbol: str, available_symbols: List[str]) -> Optional[str]:
        """
        Znajdź symbol w liście dostępnych symboli brokera
        
        Args:
            base_symbol: 'EURUSD'
            available_symbols: ['EURUSD.a', 'GBPUSD.a', 'USDJPY.a', ...]
        
        Returns:
            'EURUSD.a' (pierwszy znaleziony)
        """
        # Direct match
        if base_symbol in available_symbols:
            return base_symbol
        
        # Try all variants
        for variant in self.get_all_variants(base_symbol):
            if variant in available_symbols:
                self.logger.info(f"✅ Found {base_symbol} as {variant}")
                return variant
        
        # Fuzzy search
        base_upper = base_symbol.upper()
        for avail in available_symbols:
            if base_upper in avail.upper():
                self.logger.info(f"🔍 Fuzzy match: {base_symbol} -> {avail}")
                return avail
        
        self.logger.warning(f"⚠️ Symbol not found: {base_symbol}")
        return None
    
    def get_correlation_pair(self, base_symbol: str, available_symbols: List[str]) -> Optional[str]:
        """
        Znajdź skorelowaną parę dla hedgingu
        
        Korelacje:
        - EURUSD <-> GBPUSD (positive)
        - EURUSD <-> USDCHF (negative)
        - GBPUSD <-> EURGBP (positive)
        """
        correlation_map = {
            'EURUSD': ['GBPUSD', 'USDCHF', 'EURJPY'],
            'GBPUSD': ['EURUSD', 'EURGBP', 'GBPJPY'],
            'USDJPY': ['EURJPY', 'GBPJPY', 'AUDJPY'],
            'AUDUSD': ['NZDUSD', 'EURAUD', 'AUDJPY'],
            'USDCAD': ['EURCAD', 'USDCHF', 'CADJPY'],
            'XAUUSD': ['XAGUSD', 'EURUSD'],
        }
        
        correlated = correlation_map.get(base_symbol, [])
        
        for pair in correlated:
            found = self.find_symbol_in_list(pair, available_symbols)
            if found:
                self.logger.info(f"🔗 Correlation pair for {base_symbol}: {found}")
                return found
        
        return None
    
    def validate_symbol(self, symbol: str) -> bool:
        """Sprawdź czy symbol jest poprawny"""
        normalized = self.normalize_symbol(symbol)
        return normalized in self.ALL_BASE_SYMBOLS


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    mapper = SymbolMapper()
    
    # Test normalization
    test_symbols = [
        'EURUSD.a', 'EURUSDm', '#EURUSD', 'EURUSD_i', 'EURUSDpro',
        'GBPUSD.a', 'GBPUSDm', 'USDJPY.a',
        'GOLD', 'XAUUSD.a', '#BTCUSD'
    ]
    
    print("\n🔄 Normalization Test:")
    for sym in test_symbols:
        normalized = mapper.normalize_symbol(sym)
        print(f"  {sym:15} -> {normalized}")
    
    # Test pattern detection
    print("\n🔍 Pattern Detection:")
    pattern = mapper.detect_broker_pattern(test_symbols)
    print(f"  Detected: {pattern}")
    
    # Test batch normalize
    print("\n📦 Batch Normalize:")
    mapping = mapper.batch_normalize(test_symbols)
    for broker, base in mapping.items():
        print(f"  {broker:15} -> {base}")
    
    # Test find symbol
    print("\n🎯 Find Symbol in List:")
    available = ['EURUSD.a', 'GBPUSD.a', 'USDJPY.a', 'GOLD.a']
    found = mapper.find_symbol_in_list('EURUSD', available)
    print(f"  EURUSD -> {found}")
    
    # Test correlation
    print("\n🔗 Correlation Pair:")
    corr = mapper.get_correlation_pair('EURUSD', available)
    print(f"  EURUSD -> {corr}")