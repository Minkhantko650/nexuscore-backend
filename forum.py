from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import get_db
from models import ForumPost, ForumReply, User
from auth import get_current_user

router = APIRouter(prefix="/forum", tags=["forum"])


class PostCreate(BaseModel):
    title: str
    content: str
    category: str


class ReplyCreate(BaseModel):
    content: str


@router.get("/posts")
def get_posts(category: str = None, db: Session = Depends(get_db)):
    query = db.query(ForumPost)
    if category:
        query = query.filter(ForumPost.category == category)
    posts = query.order_by(ForumPost.created_at.desc()).all()
    return [
        {
            "id": p.id,
            "title": p.title,
            "category": p.category,
            "author": p.author.username,
            "created_at": p.created_at,
            "views": p.views,
            "reply_count": len(p.replies),
        }
        for p in posts
    ]


@router.post("/posts", status_code=201)
def create_post(
    req: PostCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    post = ForumPost(
        title=req.title,
        content=req.content,
        category=req.category,
        author_id=current_user.id,
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    return {"id": post.id, "title": post.title, "category": post.category}


@router.get("/posts/{post_id}")
def get_post(post_id: int, db: Session = Depends(get_db)):
    post = db.query(ForumPost).filter(ForumPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    post.views += 1
    db.commit()
    return {
        "id": post.id,
        "title": post.title,
        "content": post.content,
        "category": post.category,
        "author": post.author.username,
        "created_at": post.created_at,
        "views": post.views,
        "replies": [
            {
                "id": r.id,
                "content": r.content,
                "author": r.author.username,
                "created_at": r.created_at,
            }
            for r in post.replies
        ],
    }


@router.post("/posts/{post_id}/replies", status_code=201)
def create_reply(
    post_id: int,
    req: ReplyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    post = db.query(ForumPost).filter(ForumPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    reply = ForumReply(
        post_id=post_id,
        content=req.content,
        author_id=current_user.id,
    )
    db.add(reply)
    db.commit()
    db.refresh(reply)
    return {"id": reply.id, "content": reply.content}
