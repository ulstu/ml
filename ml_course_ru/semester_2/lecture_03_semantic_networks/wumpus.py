"""
Wumpus World - Пример логического вывода с пропозициональной логикой
======================================================================

Мир Wumpus - классическая задача из области искусственного интеллекта.
Агент должен найти золото и выбраться из пещеры, избегая ям и Wumpus.

Правила мира:
- Сетка 4x4
- Агент начинает в клетке (0,0)
- В мире есть Wumpus (монстр), ямы и золото
- Агент чувствует вонь (stench) рядом с Wumpus
- Агент чувствует ветерок (breeze) рядом с ямой
- Агент видит блеск (glitter) в клетке с золотом
"""

from typing import Set, Tuple, List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np


class PropositionalKB:
    """База знаний с пропозициональной логикой"""
    
    def __init__(self):
        self.facts: Set[str] = set()  # Известные факты
        self.safe_cells: Set[Tuple[int, int]] = set()
        self.wumpus_cells: Set[Tuple[int, int]] = set()
        self.pit_cells: Set[Tuple[int, int]] = set()
        self.visited_cells: Set[Tuple[int, int]] = set()
        
    def tell(self, fact: str):
        """Добавить факт в базу знаний"""
        self.facts.add(fact)
        
    def ask(self, query: str) -> bool:
        """Проверить, истинен ли факт"""
        return query in self.facts
    
    def infer_safe_cells(self, position: Tuple[int, int], percepts: Dict[str, bool]):
        """
        Логический вывод безопасных клеток на основе восприятий
        
        Правила логического вывода:
        1. Если нет ветерка, то соседние клетки не содержат ям
        2. Если нет вони, то соседние клетки не содержат Wumpus
        3. Посещенная клетка безопасна
        """
        x, y = position
        self.visited_cells.add(position)
        self.safe_cells.add(position)
        
        # Получаем соседние клетки
        neighbors = self._get_neighbors(x, y)
        
        # Если нет ветерка - соседние клетки не содержат ям
        if not percepts.get('breeze', False):
            for nx, ny in neighbors:
                self.tell(f"~Pit_{nx}_{ny}")  # ~ означает отрицание
                if (nx, ny) not in self.pit_cells:
                    self.safe_cells.add((nx, ny))
        else:
            # Есть ветерок - хотя бы одна соседняя клетка содержит яму
            self.tell(f"Breeze_{x}_{y}")
            
        # Если нет вони - соседние клетки не содержат Wumpus
        if not percepts.get('stench', False):
            for nx, ny in neighbors:
                self.tell(f"~Wumpus_{nx}_{ny}")
                if (nx, ny) not in self.wumpus_cells:
                    self.safe_cells.add((nx, ny))
        else:
            # Есть вонь - Wumpus в одной из соседних клеток
            self.tell(f"Stench_{x}_{y}")
            
        # Блеск означает золото в текущей клетке
        if percepts.get('glitter', False):
            self.tell(f"Gold_{x}_{y}")
            
    def _get_neighbors(self, x: int, y: int, size: int = 4) -> List[Tuple[int, int]]:
        """Получить соседние клетки"""
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                neighbors.append((nx, ny))
        return neighbors
    
    def mark_dangerous(self, position: Tuple[int, int], danger_type: str):
        """Пометить клетку как опасную"""
        if danger_type == 'pit':
            self.pit_cells.add(position)
            self.safe_cells.discard(position)
        elif danger_type == 'wumpus':
            self.wumpus_cells.add(position)
            self.safe_cells.discard(position)


class WumpusWorld:
    """Мир Wumpus"""
    
    def __init__(self, size: int = 4):
        self.size = size
        self.agent_pos = (0, 0)
        self.wumpus_pos = None
        self.gold_pos = None
        self.pits: Set[Tuple[int, int]] = set()
        self.wumpus_alive = True
        self.has_gold = False
        self.has_arrow = True
        
    def setup_world(self, wumpus_pos: Tuple[int, int], 
                    gold_pos: Tuple[int, int], 
                    pits: List[Tuple[int, int]]):
        """Настроить мир"""
        self.wumpus_pos = wumpus_pos
        self.gold_pos = gold_pos
        self.pits = set(pits)
        
    def get_percepts(self, position: Tuple[int, int]) -> Dict[str, bool]:
        """
        Получить восприятия агента в данной позиции
        
        Returns:
            Dict с ключами: breeze, stench, glitter, bump, scream
        """
        x, y = position
        percepts = {
            'breeze': False,
            'stench': False,
            'glitter': False,
            'bump': False,
            'scream': False
        }
        
        # Проверяем соседние клетки на ямы
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in self.pits:
                percepts['breeze'] = True
            if (nx, ny) == self.wumpus_pos and self.wumpus_alive:
                percepts['stench'] = True
                
        # Проверяем золото
        if position == self.gold_pos:
            percepts['glitter'] = True
            
        return percepts
    
    def is_safe(self, position: Tuple[int, int]) -> bool:
        """Проверить, безопасна ли клетка"""
        if position in self.pits:
            return False
        if position == self.wumpus_pos and self.wumpus_alive:
            return False
        return True


class WumpusAgent:
    """Агент с логическим выводом"""
    
    def __init__(self, world: WumpusWorld):
        self.world = world
        self.kb = PropositionalKB()
        self.position = (0, 0)
        self.path: List[Tuple[int, int]] = [(0, 0)]
        self.plan: List[Tuple[int, int]] = []
        
    def explore(self):
        """Исследовать мир используя логический вывод"""
        print(f"\n{'='*60}")
        print("НАЧАЛО ИССЛЕДОВАНИЯ МИРА WUMPUS")
        print(f"{'='*60}\n")
        
        # Начальная позиция всегда безопасна
        self.kb.safe_cells.add(self.position)
        
        while not self.world.has_gold:
            print(f"Агент в позиции: {self.position}")
            
            # Получаем восприятия
            percepts = self.world.get_percepts(self.position)
            print(f"Восприятия: {percepts}")
            
            # Обновляем базу знаний
            self.kb.infer_safe_cells(self.position, percepts)
            
            # Логический вывод
            self._print_logical_inference(percepts)
            
            # Если нашли золото - берем его
            if percepts['glitter']:
                self.world.has_gold = True
                print("\n🏆 ЗОЛОТО НАЙДЕНО!")
                break
                
            # Выбираем следующий ход
            next_pos = self._choose_next_move()
            if next_pos is None:
                print("\n❌ Нет безопасных ходов!")
                break
                
            self.position = next_pos
            self.path.append(next_pos)
            print(f"➡️  Переход в клетку: {next_pos}\n")
            print("-" * 60)
            
    def _print_logical_inference(self, percepts: Dict[str, bool]):
        """Вывести логические заключения"""
        print("\n🧠 ЛОГИЧЕСКИЙ ВЫВОД:")
        x, y = self.position
        
        neighbors = self.kb._get_neighbors(x, y)
        
        if not percepts['breeze']:
            print(f"  ✓ Нет ветерка → соседние клетки {neighbors} НЕ содержат ям")
            for nx, ny in neighbors:
                print(f"    Правило: ~Breeze_{x}_{y} → ~Pit_{nx}_{ny}")
        else:
            print(f"  ⚠ Есть ветерок → хотя бы одна из клеток {neighbors} содержит яму")
            print(f"    Правило: Breeze_{x}_{y} → (Pit_{neighbors[0]} ∨ Pit_{neighbors[1]} ∨ ...)")
            
        if not percepts['stench']:
            print(f"  ✓ Нет вони → соседние клетки {neighbors} НЕ содержат Wumpus")
            for nx, ny in neighbors:
                print(f"    Правило: ~Stench_{x}_{y} → ~Wumpus_{nx}_{ny}")
        else:
            print(f"  ⚠ Есть вонь → Wumpus в одной из клеток {neighbors}")
            print(f"    Правило: Stench_{x}_{y} → (Wumpus_{neighbors[0]} ∨ Wumpus_{neighbors[1]} ∨ ...)")
            
        print(f"\n  Безопасные клетки: {sorted(self.kb.safe_cells)}")
        print(f"  Посещенные клетки: {sorted(self.kb.visited_cells)}")
        
    def _choose_next_move(self) -> Tuple[int, int]:
        """Выбрать следующий ход"""
        # Ищем безопасные непосещенные клетки среди соседних
        x, y = self.position
        candidates = []
        
        for nx, ny in self.kb._get_neighbors(x, y):
            if (nx, ny) in self.kb.safe_cells and (nx, ny) not in self.kb.visited_cells:
                candidates.append((nx, ny))
                
        if candidates:
            return candidates[0]
        
        # Если нет новых безопасных клеток, ищем путь к непосещенным безопасным
        for safe_cell in self.kb.safe_cells:
            if safe_cell not in self.kb.visited_cells:
                return self._find_path_to(safe_cell)
                
        return None
    
    def _find_path_to(self, target: Tuple[int, int]) -> Tuple[int, int]:
        """Найти путь к целевой клетке (простой BFS)"""
        from collections import deque
        
        queue = deque([self.position])
        visited = {self.position}
        parent = {self.position: None}
        
        while queue:
            current = queue.popleft()
            
            if current == target:
                # Восстанавливаем путь
                path = []
                while parent[current] is not None:
                    path.append(current)
                    current = parent[current]
                return path[-1] if path else None
                
            for neighbor in self.kb._get_neighbors(*current):
                if neighbor in self.kb.safe_cells and neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
                    
        return None


class WumpusVisualizer:
    """Визуализация мира Wumpus"""
    
    def __init__(self, world: WumpusWorld, agent: WumpusAgent):
        self.world = world
        self.agent = agent
        
    def visualize(self, title: str = "Wumpus World"):
        """Создать визуализацию"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Левая панель - реальный мир
        self._draw_actual_world(ax1)
        
        # Правая панель - знания агента
        self._draw_agent_knowledge(ax2)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def _draw_actual_world(self, ax):
        """Нарисовать реальный мир"""
        ax.set_xlim(-0.5, self.world.size - 0.5)
        ax.set_ylim(-0.5, self.world.size - 0.5)
        ax.set_aspect('equal')
        ax.set_title('Реальный мир (скрыт от агента)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(self.world.size))
        ax.set_yticks(range(self.world.size))
        ax.grid(True, linewidth=2)
        ax.invert_yaxis()
        
        # Рисуем клетки
        for x in range(self.world.size):
            for y in range(self.world.size):
                # Ямы
                if (x, y) in self.world.pits:
                    circle = plt.Circle((x, y), 0.3, color='black', alpha=0.7)
                    ax.add_patch(circle)
                    ax.text(x, y, '⚫\nПит', ha='center', va='center', 
                           fontsize=10, color='white', fontweight='bold')
                
                # Wumpus
                if (x, y) == self.world.wumpus_pos:
                    rect = FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8,
                                         boxstyle="round,pad=0.05",
                                         edgecolor='red', facecolor='darkred', 
                                         linewidth=3, alpha=0.7)
                    ax.add_patch(rect)
                    ax.text(x, y, '👹\nWumpus', ha='center', va='center',
                           fontsize=10, color='white', fontweight='bold')
                
                # Золото
                if (x, y) == self.world.gold_pos:
                    star = plt.scatter(x, y, s=1000, c='gold', marker='*', 
                                      edgecolors='orange', linewidths=2, zorder=5)
                    ax.text(x, y+0.25, '💰', ha='center', va='center', fontsize=20)
        
        # Рисуем путь агента
        if len(self.agent.path) > 1:
            path_x = [p[0] for p in self.agent.path]
            path_y = [p[1] for p in self.agent.path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.5, label='Путь агента')
            
        # Текущая позиция агента
        ax.scatter(self.agent.position[0], self.agent.position[1], 
                  s=500, c='blue', marker='o', edgecolors='darkblue', 
                  linewidths=3, zorder=10, label='Агент')
        ax.text(self.agent.position[0], self.agent.position[1], '🤖', 
               ha='center', va='center', fontsize=16)
        
        ax.legend(loc='upper right')
        
    def _draw_agent_knowledge(self, ax):
        """Нарисовать знания агента"""
        ax.set_xlim(-0.5, self.world.size - 0.5)
        ax.set_ylim(-0.5, self.world.size - 0.5)
        ax.set_aspect('equal')
        ax.set_title('Знания агента (логический вывод)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(self.world.size))
        ax.set_yticks(range(self.world.size))
        ax.grid(True, linewidth=2)
        ax.invert_yaxis()
        
        # Рисуем клетки по знаниям агента
        for x in range(self.world.size):
            for y in range(self.world.size):
                if (x, y) in self.agent.kb.visited_cells:
                    # Посещенная клетка
                    rect = FancyBboxPatch((x-0.45, y-0.45), 0.9, 0.9,
                                         boxstyle="round,pad=0.02",
                                         edgecolor='green', facecolor='lightgreen',
                                         linewidth=2, alpha=0.3)
                    ax.add_patch(rect)
                    ax.text(x, y-0.35, 'Посещена', ha='center', va='center',
                           fontsize=7, color='green', fontweight='bold')
                    
                elif (x, y) in self.agent.kb.safe_cells:
                    # Безопасная непосещенная клетка
                    rect = FancyBboxPatch((x-0.45, y-0.45), 0.9, 0.9,
                                         boxstyle="round,pad=0.02",
                                         edgecolor='blue', facecolor='lightblue',
                                         linewidth=2, alpha=0.3)
                    ax.add_patch(rect)
                    ax.text(x, y-0.35, 'Безопасно', ha='center', va='center',
                           fontsize=7, color='blue', fontweight='bold')
                    
                if (x, y) in self.agent.kb.pit_cells:
                    # Возможная яма
                    ax.text(x, y, '⚠️\nЯма?', ha='center', va='center',
                           fontsize=10, color='red', fontweight='bold')
                    
                if (x, y) in self.agent.kb.wumpus_cells:
                    # Возможный Wumpus
                    ax.text(x, y, '👹\nWumpus?', ha='center', va='center',
                           fontsize=10, color='darkred', fontweight='bold')
                    
        # Текущая позиция агента
        ax.scatter(self.agent.position[0], self.agent.position[1],
                  s=500, c='blue', marker='o', edgecolors='darkblue',
                  linewidths=3, zorder=10)
        ax.text(self.agent.position[0], self.agent.position[1], '🤖',
               ha='center', va='center', fontsize=16)
        
        # Легенда
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', edgecolor='green', label='Посещено'),
            Patch(facecolor='lightblue', edgecolor='blue', label='Безопасно'),
            Patch(facecolor='white', edgecolor='black', label='Неизвестно')
        ]
        ax.legend(handles=legend_elements, loc='upper right')


def main():
    """Главная функция - демонстрация работы"""
    print("\n" + "="*60)
    print("WUMPUS WORLD - ЛОГИЧЕСКИЙ ВЫВОД С ПРОПОЗИЦИОНАЛЬНОЙ ЛОГИКОЙ")
    print("="*60)
    
    # Создаем мир
    world = WumpusWorld(size=4)
    
    # Настраиваем мир (простой пример)
    world.setup_world(
        wumpus_pos=(1, 3),      # Wumpus в клетке (1, 3)
        gold_pos=(2, 3),         # Золото в клетке (2, 3)
        pits=[(3, 1), (3, 3), (1, 2)]  # Ямы в клетках
    )
    
    print("\nНастройка мира:")
    print(f"  Wumpus: {world.wumpus_pos}")
    print(f"  Золото: {world.gold_pos}")
    print(f"  Ямы: {sorted(world.pits)}")
    
    # Создаем агента
    agent = WumpusAgent(world)
    
    # Агент исследует мир
    agent.explore()
    
    # Визуализация
    print(f"\n{'='*60}")
    print("ВИЗУАЛИЗАЦИЯ")
    print(f"{'='*60}\n")
    
    visualizer = WumpusVisualizer(world, agent)
    visualizer.visualize("Wumpus World - Результат исследования")
    
    # Итоговая статистика
    print(f"\n{'='*60}")
    print("ИТОГИ")
    print(f"{'='*60}")
    print(f"Золото найдено: {'✓ Да' if world.has_gold else '✗ Нет'}")
    print(f"Посещено клеток: {len(agent.kb.visited_cells)}")
    print(f"Путь агента: {agent.path}")
    print(f"Безопасных клеток найдено: {len(agent.kb.safe_cells)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
