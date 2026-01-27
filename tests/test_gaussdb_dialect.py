import unittest
import logging
from sqlglot import parse_one

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GaussDBDialectTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def test_create_table(self):
        sql = """
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            user_name VARCHAR(50) NOT NULL,
            email VARCHAR(100) UNIQUE,
            age INT CHECK (age >= 0),
            active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[CREATE TABLE]\n{repr(ob_ast)}")

    def test_create_index(self):
        # 普通索引
        sql = "CREATE INDEX idx_users_email ON users (email)"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[CREATE INDEX]\n{repr(ob_ast)}")

        # 唯一索引
        sql = "CREATE UNIQUE INDEX uk_users_user_name ON users (user_name)"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[CREATE UNIQUE INDEX]\n{repr(ob_ast)}")

    def test_insert(self):
        sql = "INSERT INTO users (user_name, email, age) VALUES ('alice', 'alice@example.com', 30)"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[INSERT]\n{repr(ob_ast)}")

        sql = "INSERT INTO users (user_name, email) VALUES ('bob', 'bob@example.com'), ('charlie', 'char@example.com')"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[BATCH INSERT]\n{repr(ob_ast)}")

    def test_select(self):
        sql = "SELECT id, user_name, email FROM users WHERE active = true ORDER BY id LIMIT 10"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[SELECT]\n{repr(ob_ast)}")

    def test_update(self):
        sql = "UPDATE users SET email = 'new@example.com', updated_at = NOW() WHERE id = 1"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[UPDATE]\n{repr(ob_ast)}")

    def test_delete(self):
        sql = "DELETE FROM users WHERE age < 18"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[DELETE ROWS]\n{repr(ob_ast)}")

    def test_alter_table_operations(self):
        # 添加列
        sql = "ALTER TABLE users ADD COLUMN phone VARCHAR(20) DEFAULT ''"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[ADD COLUMN]\n{repr(ob_ast)}")

        # 删除列
        sql = "ALTER TABLE users DROP COLUMN age"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[DROP COLUMN]\n{repr(ob_ast)}")

        # 重命名列
        sql = "ALTER TABLE users RENAME COLUMN user_name TO username"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[RENAME COLUMN]\n{repr(ob_ast)}")

        # 修改列类型
        sql = "ALTER TABLE users ALTER COLUMN email TYPE VARCHAR(150)"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[ALTER COLUMN TYPE]\n{repr(ob_ast)}")

        # 设置非空
        sql = "ALTER TABLE users ALTER COLUMN username SET NOT NULL"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[SET NOT NULL]\n{repr(ob_ast)}")

        # 设置默认值
        sql = "ALTER TABLE users ALTER COLUMN phone SET DEFAULT 'unknown'"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(ob_ast)
        logger.info(f"\n[SET DEFAULT]\n{repr(ob_ast)}")

    def test_drop_table(self):
        # 删除表（带 IF EXISTS 更安全）
        sql = "DROP TABLE IF EXISTS users"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[DROP TABLE]\n{repr(ob_ast)}")

        # 直接删除（无 IF EXISTS）
        sql = "DROP TABLE users"
        ob_ast = parse_one(sql, dialect="postgres")
        logger.info(f"\n[DROP TABLE STRICT]\n{repr(ob_ast)}")