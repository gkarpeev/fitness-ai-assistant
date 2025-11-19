"""Main entry point for the AI Fitness Assistant."""
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag.vectorstore import FitnessVectorStore, initialize_vectorstore_with_data
from src.agents.coordinator import FitnessCoordinator
from src.models.schemas import UserProfile, FitnessGoal, ExperienceLevel, EquipmentAccess
import config

console = Console()


def initialize_system():
    """Initialize the fitness assistant system."""
    console.print("\n[bold cyan]🏋️ AI Fitness Assistant[/bold cyan]")
    console.print("[cyan]Инициализация системы...[/cyan]\n")
    
    # Check if vectorstore exists
    if not (config.CHROMA_DIR / "chroma.sqlite3").exists():
        console.print("[yellow]📦 Первый запуск - создание базы знаний...[/yellow]")
        vectorstore = initialize_vectorstore_with_data()
    else:
        console.print("[green]✅ Загрузка существующей базы знаний...[/green]")
        vectorstore = FitnessVectorStore()
    
    # Initialize coordinator
    console.print("\n[cyan]🤖 Инициализация агентов...[/cyan]")
    coordinator = FitnessCoordinator(vectorstore)
    
    console.print("\n[bold green]✅ Система готова к работе![/bold green]\n")
    
    return coordinator


def create_sample_user():
    """Create a sample user profile."""
    return UserProfile(
        user_id="user_001",
        age=28,
        weight=75,
        height=180,
        gender="male",
        goal=FitnessGoal.MUSCLE_GAIN,
        experience_level=ExperienceLevel.INTERMEDIATE,
        equipment_access=EquipmentAccess.GYM,
        injuries=[],
        dietary_restrictions=[]
    )


def interactive_mode(coordinator: FitnessCoordinator):
    """Run interactive chat mode."""
    console.print(Panel.fit(
        "[bold cyan]AI Fitness Assistant - Интерактивный режим[/bold cyan]\n\n"
        "Команды:\n"
        "  [yellow]exit[/yellow] - выход\n"
        "  [yellow]new user[/yellow] - создать профиль пользователя\n"
        "  [yellow]workout[/yellow] - вопрос агенту тренировок\n"
        "  [yellow]nutrition[/yellow] - вопрос агенту питания\n"
        "  [yellow]both[/yellow] - комплексный запрос\n\n"
        "Или просто задайте вопрос!",
        title="Добро пожаловать"
    ))
    
    current_user_id = None
    
    while True:
        try:
            console.print("\n[bold cyan]Вы:[/bold cyan] ", end="")
            user_input = input().strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == "exit":
                console.print("\n[yellow]👋 До свидания![/yellow]\n")
                break
            
            elif user_input.lower() == "new user":
                # Create sample user (в реальной системе здесь был бы ввод данных)
                user = create_sample_user()
                coordinator.register_user(user)
                current_user_id = user.user_id
                console.print(f"\n[green]✅ Создан пользователь: {user.user_id}[/green]")
                console.print(f"   Цель: {user.goal.value}")
                console.print(f"   Уровень: {user.experience_level.value}")
                continue
            
            # Determine agent routing
            force_agent = None
            if user_input.lower().startswith("workout:"):
                force_agent = "workout"
                user_input = user_input[8:].strip()
            elif user_input.lower().startswith("nutrition:"):
                force_agent = "nutrition"
                user_input = user_input[10:].strip()
            elif user_input.lower().startswith("both:"):
                force_agent = "both"
                user_input = user_input[5:].strip()
            
            # Process query
            console.print("\n[cyan]💭 Обрабатываю запрос...[/cyan]\n")
            
            result = coordinator.process_query(
                user_message=user_input,
                user_id=current_user_id,
                force_agent=force_agent
            )
            
            # Display response
            console.print("\n[bold green]🤖 Ассистент:[/bold green]\n")
            
            # Render as markdown for better formatting
            md = Markdown(result["response"])
            console.print(md)
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]👋 До свидания![/yellow]\n")
            break
        except Exception as e:
            console.print(f"\n[red]❌ Ошибка: {e}[/red]\n")


def demo_mode(coordinator: FitnessCoordinator):
    """Run demo with predefined queries."""
    console.print(Panel.fit(
        "[bold cyan]AI Fitness Assistant - Демо режим[/bold cyan]\n\n"
        "Демонстрация возможностей системы с примерами запросов",
        title="Демонстрация"
    ))
    
    # Create sample user
    user = create_sample_user()
    coordinator.register_user(user)
    
    # Demo queries
    demo_queries = [
        {
            "title": "Программа тренировок для набора массы",
            "query": "Создай программу тренировок на 3 дня в неделю для набора мышечной массы",
            "agent": "workout"
        },
        {
            "title": "Рекомендации по добавкам",
            "query": "Какие добавки мне нужны для набора массы? У меня средний бюджет.",
            "agent": "nutrition"
        },
        {
            "title": "План питания",
            "query": "Рассчитай мне калории и составь примерный план питания",
            "agent": "nutrition"
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold yellow]Демо {i}/{len(demo_queries)}: {demo['title']}[/bold yellow]")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]\n")
        
        console.print(f"[cyan]Запрос:[/cyan] {demo['query']}\n")
        console.print("[cyan]💭 Обрабатываю...[/cyan]\n")
        
        result = coordinator.process_query(
            user_message=demo['query'],
            user_id=user.user_id,
            force_agent=demo['agent']
        )
        
        console.print("[bold green]🤖 Ответ:[/bold green]\n")
        md = Markdown(result["response"])
        console.print(md)
        
        if i < len(demo_queries):
            input("\n[yellow]Нажмите Enter для следующего примера...[/yellow]")


def main():
    """Main function."""
    # Initialize system
    coordinator = initialize_system()
    
    # Choose mode
    console.print("[bold]Выберите режим:[/bold]")
    console.print("  1. Интерактивный режим (чат)")
    console.print("  2. Демо режим (примеры)")
    console.print("\n[cyan]Введите номер (1 или 2):[/cyan] ", end="")
    
    choice = input().strip()
    
    if choice == "2":
        demo_mode(coordinator)
    else:
        interactive_mode(coordinator)


if __name__ == "__main__":
    main()
