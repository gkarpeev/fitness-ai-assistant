"""Script to initialize or reset the knowledge base."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from src.rag.vectorstore import initialize_vectorstore_with_data
import config

console = Console()


def main():
    """Initialize or reset the vector store database."""
    console.print("\n[bold cyan]🔄 Database Initialization[/bold cyan]")
    console.print("[cyan]=" * 60 + "[/cyan]\n")
    
    # Check if database already exists
    db_exists = (config.CHROMA_DIR / "chroma.sqlite3").exists()
    
    if db_exists:
        console.print("[yellow]⚠️  База данных уже существует![/yellow]")
        console.print("\nОпции:")
        console.print("  1. Пересоздать базу (удалить и создать заново)")
        console.print("  2. Отмена")
        
        choice = console.input("\n[cyan]Выберите опцию (1 или 2):[/cyan] ")
        
        if choice != "1":
            console.print("\n[yellow]Операция отменена[/yellow]\n")
            return
        
        # Delete existing database
        console.print("\n[yellow]🗑️  Удаление старой базы данных...[/yellow]")
        import shutil
        try:
            shutil.rmtree(config.CHROMA_DIR)
            config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            console.print("[green]✅ Старая база удалена[/green]")
        except Exception as e:
            console.print(f"[red]❌ Ошибка удаления: {e}[/red]")
            return
    
    # Initialize database
    console.print("\n[cyan]🚀 Создание новой базы данных...[/cyan]\n")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Инициализация...", total=None)
            vectorstore = initialize_vectorstore_with_data()
            progress.update(task, completed=True)
        
        console.print("\n[bold green]✅ База данных успешно создана![/bold green]")
        console.print(f"[green]📍 Расположение: {config.CHROMA_DIR}[/green]")
        
        # Show statistics
        console.print("\n[cyan]📊 Статистика базы знаний:[/cyan]")
        
        try:
            exercises_count = len(vectorstore.collections["exercises"].get()["ids"])
            supplements_count = len(vectorstore.collections["supplements"].get()["ids"])
            nutrition_count = len(vectorstore.collections["nutrition"].get()["ids"])
            
            console.print(f"  • Упражнения: [bold]{exercises_count}[/bold]")
            console.print(f"  • Добавки: [bold]{supplements_count}[/bold]")
            console.print(f"  • Статьи о питании: [bold]{nutrition_count}[/bold]")
            console.print(f"  • Всего документов: [bold]{exercises_count + supplements_count + nutrition_count}[/bold]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Не удалось получить статистику: {e}[/yellow]")
        
        console.print("\n[green]🎉 Готово! Теперь можете запустить main.py[/green]\n")
        
    except Exception as e:
        console.print(f"\n[red]❌ Ошибка инициализации: {e}[/red]\n")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")


if __name__ == "__main__":
    main()