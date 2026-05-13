import asyncio
from app.tools.food_tools import query_food


async def main():
    result = await query_food.ainvoke({
        "destination": "北京",
        "food_type": "restaurant",
    })
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
