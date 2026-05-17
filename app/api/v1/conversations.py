"""会话管理路由：CRUD 5 个端点"""
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.conversation import ConversationCreate, ConversationUpdate, ConversationResponse
from app.api.v1.deps import get_db
from app.utils.logger import app_logger

router = APIRouter(prefix="/conversations", tags=["会话"])


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(body: ConversationCreate, pool_user: tuple = Depends(get_db)):
    """创建新会话"""
    pool, user_id = pool_user
    conv_id = uuid4()

    async with pool.connection() as conn:
        row = await (
            await conn.execute(
                """
                INSERT INTO conversations (id, user_id, title, current_model, system_prompt)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING *
                """,
                (conv_id, user_id, body.title, body.current_model, body.system_prompt),
            )
        ).fetchone()

    app_logger.info(f"会话创建: {conv_id} (user={user_id})")
    return dict(row)


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    pool_user: tuple = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """获取会话列表"""
    pool, user_id = pool_user
    async with pool.connection() as conn:
        rows = await (
            await conn.execute(
                """
                SELECT * FROM conversations
                WHERE user_id = %s AND status != 'deleted'
                ORDER BY updated_at DESC
                LIMIT %s OFFSET %s
                """,
                (user_id, limit, offset),
            )
        ).fetchall()
    return [dict(r) for r in rows]


@router.get("/{conv_id}", response_model=ConversationResponse)
async def get_conversation(conv_id: str, pool_user: tuple = Depends(get_db)):
    """获取会话详情"""
    pool, user_id = pool_user
    async with pool.connection() as conn:
        row = await (
            await conn.execute(
                "SELECT * FROM conversations WHERE id = %s", (conv_id,)
            )
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")

    conv = dict(row)
    if str(conv["user_id"]) != str(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问")

    return conv


@router.patch("/{conv_id}", response_model=ConversationResponse)
async def update_conversation(
    conv_id: str,
    body: ConversationUpdate,
    pool_user: tuple = Depends(get_db),
):
    """更新会话（归属校验）"""
    pool, user_id = pool_user

    async with pool.connection() as conn:
        existing = await (
            await conn.execute(
                "SELECT * FROM conversations WHERE id = %s", (conv_id,)
            )
        ).fetchone()
        if existing is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")

        conv = dict(existing)
        if str(conv["user_id"]) != str(user_id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问")

        updates = body.model_dump(exclude_none=True)
        if not updates:
            return conv

        set_clauses = []
        params = []
        for i, (key, val) in enumerate(updates.items(), start=1):
            set_clauses.append(f"{key} = %s")
            params.append(val)

        params.append(conv_id)
        conv_id_idx = len(params)

        sql = (
            f"UPDATE conversations SET {', '.join(set_clauses)}, updated_at = NOW() "
            f"WHERE id = %s RETURNING *"
        )
        row = await (await conn.execute(sql, tuple(params))).fetchone()

    return dict(row)


@router.delete("/{conv_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conv_id: str, pool_user: tuple = Depends(get_db)):
    """软删除会话"""
    pool, user_id = pool_user

    async with pool.connection() as conn:
        existing = await (
            await conn.execute(
                "SELECT * FROM conversations WHERE id = %s", (conv_id,)
            )
        ).fetchone()
        if existing is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
        if str(dict(existing)["user_id"]) != str(user_id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问")

        await conn.execute(
            "UPDATE conversations SET status = 'deleted', updated_at = NOW() WHERE id = %s",
            (conv_id,),
        )

    app_logger.info(f"会话已删除: {conv_id}")
