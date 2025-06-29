CREATE USER grafana_user WITH PASSWORD 'postgres_grafana_pwd';

DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'postgres') THEN
      CREATE DATABASE postgres;
      GRANT ALL PRIVILEGES ON DATABASE postgres TO grafana_user;
   END IF;
END
$$;
