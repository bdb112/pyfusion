"""Object relational mapping for Pyfusion"""

import pyfusion

if pyfusion.USE_ORM:
    import sqlalchemy

def setup_orm():
    from sqlalchemy import create_engine, MetaData
    from sqlalchemy.orm import scoped_session, sessionmaker
    from sqlalchemy.ext.declarative import declarative_base

    pyfusion.orm_engine = create_engine(pyfusion.config.get('global', 'database'))
    
    pyfusion.Session = scoped_session(sessionmaker(autocommit=False,
                                      autoflush=True,
                                      bind=pyfusion.orm_engine))

    pyfusion.metadata = MetaData()
    pyfusion.metadata.bind = pyfusion.orm_engine
    #pyfusion.Base = declarative_base(bind=pyfusion.orm_engine)
