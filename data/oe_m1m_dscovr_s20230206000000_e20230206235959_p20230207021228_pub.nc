CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230206000000_e20230206235959_p20230207021228_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-07T02:12:28.526Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-06T00:00:00.000Z   time_coverage_end         2023-02-06T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        e   records_fill         ;   records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxb@[�  
�          A�R��Q�@�(��p��~�B�G���Q�@���(��G�B�B�                                    Bxb@j&  �          A#��p�@0����
aHC��p�@�����{�3�RB��f                                    Bxb@x�  
�          A �׿c�
@>�R����HB�B��c�
@߮��z��5p�B                                    Bxb@�r  
Z          A?�ff@Tz��Q�� B�Ǯ?�ff@ڏ\�\�#�B�\)                                    Bxb@�  
�          A  @%@S�
�\)�~\)BO�H@%@ٙ�������RB�G�                                    Bxb@��  
�          A��?�
=@#33�G�G�Br��?�
=@������:G�B���                                    Bxb@�d  T          A�?�{?޸R�G�\)Be�\?�{@�p��陚�Np�B���                                    Bxb@�
            A��{?z�H���§��B��
��{@����dQ�B�aH                                    Bxb@а  �          A z�>8Q�?�G��G���B��>8Q�@������R�Sp�B���                                    Bxb@�V  
�          A�\���
?����z�¦  B��׽��
@�Q���R�`{B���                                    Bxb@��  
�          A{>��h����¨�C�U�>�@h����RW
B���                                    Bxb@��  �          A33?W
=��\)�z�£��C�j=?W
=@\(���\B�u�                                    BxbAH  
�          A��?����Q����\C�+�?��@5����ǮB�Q�                                    BxbA�  
�          A�H?�(��Y����¢C���?�(�@l(��{�)B�aH                                    BxbA(�  �          A\)?333��(���Rª�=C�%?333@���z�z�B��R                                    BxbA7:  T          AG�?�{�ٙ��G��3C�XR?�{@-p��p��B}(�                                    BxbAE�  7          A\)@%����(��C�L�@%�@33���Bp�                                    BxbAT�  ?          A�H?�Q쿯\)�z��{C��=?�Q�@>�R�ff�
Bq�                                    BxbAc,  
�          A!�=�Q�>�(��   ®�B�u�=�Q�@�����rz�B���                                    BxbAq�  
Z          A(  ��\)@ ���\)#�B�(���\)@ə���
=�N�RB�W
                                    BxbA�x  �          A!p���
=A��G���Q�B��H��
=Aff�(��l(�B�{                                    BxbA�  
�          A6=q�
=qA)���w���{B��H�
=qA333?��@ڏ\B��\                                    BxbA��  
�          A:�H=��
A4���9���fffB�#�=��
A6=q@!G�AG\)B�(�                                    BxbA�j  "          AC\)��z�A=p��>{�a��B��
��z�A>ff@,(�AK�
B���                                    BxbA�  T          AC33���A>=q�*�H�J�\B��q���A=�@>�RAbffB�Ǯ                                    BxbAɶ  �          AA녿(�A<���*�H�K�
B��f�(�A;�
@<(�A`��B��                                    BxbA�\  q          AA�\)A<���)���J=qB�ff�\)A;�@=p�Ab=qB�k�                                    BxbA�  T          A@z��A<(�����<Q�B�{��A9@G
=Ap  B�(�                                    BxbA��  "          AA녾���A>�\���!p�B��;���A9p�@^�RA��B��f                                    BxbBN  T          A@(����A<���
=q�%��B��{���A8(�@X��A�33B��                                    BxbB�  T          AA=�A>=q�\)�*�RB��f=�A:{@VffA��RB��)                                    BxbB!�  "          A@��?�A>�\��z���Q�B�  ?�A4z�@�33A��B��R                                    BxbB0@  �          A<z�?p��A;\)����+�B���?p��A+33@���A�B��                                    BxbB>�  
�          A;33?}p�A9녿(���N�RB�L�?}p�A*�H@�z�A��B�p�                                    BxbBM�  
>          A9�?�RA8zᾏ\)��33B��H?�RA&�H@�A�(�B�8R                                    BxbB\2  �          A.{>�  A,  ���R���HB�L�>�  A"�\@n{A��RB�#�                                    BxbBj�  T          A2ff>�Q�A1��?B�\@z�HB�=q>�Q�AG�@�p�A���B���                                    BxbBy~  T          A"�H>��
A"ff���Ϳz�B�aH>��
A�@��RAЏ\B���                                    BxbB�$  �          Az�=�Ap��z��Mp�B�p�=�A��@�RA\��B�p�                                    BxbB��  T          A  �W
=A��������B��H�W
=A녿+���  B�z�                                    BxbB�p  "          A#\)����A���Q��  Bų3����A �׿�����(�B���                                    BxbB�  �          A-���=qA�������
B˔{��=qA(�׿����Bǽq                                    BxbB¼  T          A#
=�k�A=q�I����  B�녿k�A z�?�  A�B��                                     BxbB�b  "          A�����A�\��H�`��B�Ǯ����A�@
=qAG�
B��q                                    BxbB�  T          A��=#�
A��p��:�RB��=#�
A�@!G�AnffB��                                    BxbB�  �          A$���G�@�������B����G�A�������.�HB�                                      BxbB�T  �          A%녿��@���ə���B�ff���Ap��Q��B{B�Q�                                    BxbC�  �          A*{>#�
@�z�����6z�B��\>#�
A�
�]p����HB�33                                    BxbC�  �          A&�H>�{@�{��G��1��B��3>�{A�\�Mp���\)B�                                    BxbC)F  �          A)�?J=q@��\��Q��&B�k�?J=qA"=q�1��t��B�
=                                    BxbC7�  �          A*�H?��@����EB�{?��A������Q�B�                                      BxbCF�  �          A)G�>8Q�@�G���H(�B���>8Q�A
=��\)���RB��H                                    BxbCU8  �          A(��>���@ҏ\�Q��M�HB�B�>���A���\)��ffB�\                                    BxbCc�  �          A
=>\)@\(��ff�qB�G�>\)@�������+�
B��q                                    BxbCr�  �          A33=�\)@p���G���B���=�\)@陚�Ǯ�"
=B���                                    BxbC�*  �          A%G��\@Z=q�  8RB���\@�(���
=�133B��H                                    BxbC��  �          A%G��.{@Q��z��HB�
=�.{@�������4\)B��R                                    BxbC�v  �          A&�H��{@I����H��B�Q쾮{@�\)��  �8�B�aH                                    BxbC�  "          A'��8Q�@7��!�.B�G��8Q�@�G���Q��?�B���                                    BxbC��  "          @�>�
=?�Q����R�B�� >�
=@��R�ȣ��S�HB���                                    BxbC�h  
�          @˅?
=@�z���R��B��?
=@��L�Ϳ!G�B�Q�                                    BxbC�  
�          @xQ�?��H?��Ϳ��}BS?��H?�Q�>L��@�BY�                                    BxbC�  �          @[�?^�R������<��C��H?^�R�E��C33B�C�h�                                    BxbC�Z  T          @���?���k����H�=C�4{?��?���
=��BW�H                                    BxbD   
�          @�(�?�33�(������ C��?�33?���G��p�B5�
                                    BxbD�  "          @���@;����ff� Q�C���@;���ff�HQ��4��C�"�                                    BxbD"L  �          @�{@S�
�=p������C��@S�
�&ff��z���Q�C��                                    BxbD0�            @��@xQ���|���,Q�C�U�@xQ�>u��G��?�@b�\                                    BxbD?�  >          @�Q�@\(��G����\�=G�C��f@\(�=�Q������X�?\                                    BxbDN>  T          @�p�@Mp�����\�U33C���@Mp�>�  ����t
=@��                                    BxbD\�  �          @ʏ\@W��*�H��=q�;
=C��3@W��\�����e{C��                                     BxbDk�  �          @�z�@W��&ff��p��>z�C�(�@W���z����H�f�C��{                                    BxbDz0  T          @���@�\��R��
=�i��C��@�\>�z���p���@�p�                                    BxbD��  "          @���?����R��ff�*�RC��?���\)�N�R�Q�C�:�                                    BxbD�|  �          @�Q�@=p����R����C��\@=p��J=q�k��!G�C���                                    BxbD�"  �          @���@��R�G��G����C�ٚ@��R��(��˅�{33C�u�                                    BxbD��  
�          @�33?��R�X����33�FQ�C���?��R���������C�=q                                    BxbD�n  �          @�=q?B�\��������6=qC�t{?B�\�
�H��z��RC���                                    BxbD�  "          @ۅ?c�
��33��z��P�\C�)?c�
��=q��z���C��                                    BxbD�  �          @�?���p���
=�Bz�C�>�?��������C��f                                    BxbD�`  �          A (�@9���������R�6��C��f@9����ff��
=k�C�q                                    BxbD�  �          A
=?��������ۅ�P��C�|)?��׿����Ru�C��                                    BxbE�  
�          A�?Ǯ�������XG�C�?Ǯ���\��C�e                                    BxbER  
�          @��?˅��=q�����C�N?˅����1����
C��R                                    BxbE)�  �          @��@6ff�vff��p����
C�>�@6ff�&ff�]p��'�C���                                    BxbE8�  �          @�{?}p����׿�  �[�
C��
?}p��~{�S�
��RC�H                                    BxbEGD  T          @�  @(���?�p�A�\)C�j=@(���33�\)���HC�W
                                    BxbEU�  �          @��@#33��  ?��@ʏ\C�R@#33��G����
���\C��                                    BxbEd�  T          @�ff?��H���׿����Q�C�R?��H��G��+�����C�e                                    BxbEs6  �          @�@���~{?\A���C�}q@�������33���C��                                    BxbE��  �          @�\)@ff���H?���A���C��\@ff��녾���z�C�H�                                    BxbE��  �          @�33@\���\(��%���
C��@\�Ϳ�p��tz��/z�C��                                    BxbE�(  T          A	�@���%���Q��;�
C���@��>\��Q��P�@��
                                    BxbE��  "          A�H@����z����H�M�C��@��?�(��ʏ\�M(�A�=q                                    BxbE�t  T          A
=@������H���'�C��@������R��z��Y  C��q                                    BxbE�  
�          A!�@���������$z�C�>�@����R�����C���                                    BxbE��  "          A$z�@�\)�z�\��
C��3@�\)����Q����C�"�                                    BxbE�f  �          A ��@�p��
�\��{����C��@�p������R��=qC��                                    BxbE�  �          A#33@��\��\��z��љ�C�=q@��\��G���(�����C�S3                                    BxbF�  �          A#33@�(��
=��\�7
=C��q@�(��G������
=C�:�                                    BxbFX  T          A$(�@����{=L��>�z�C��3@�����
�~�R����C�
=                                    BxbF"�  	�          A�H@�G��  ?s33@���C��@�G�����@  ����C���                                    BxbF1�  �          A33@��R�33?(�@a�C��R@��R����Q����HC�aH                                    BxbF@J  T          Az�?�Q����
@��A��RC�#�?�Q���\?��\@�z�C�"�                                    BxbFN�  �          A
=@|�����H@>�RA��C�\)@|����(����33C���                                    BxbF]�  
�          A�@�p���
=�ff�yp�C�aH@�p�������p��p�C���                                    BxbFl<  T          AQ�@��\�������o�
C��)@��\��  �����C��\                                    BxbFz�  
�          A�
@%��
=?���Ap�C�B�@%�������r{C�t{                                    BxbF��  �          A{@����
?z�H@��C��@������.�R��{C�s3                                    BxbF�.  
�          A#\)@E���@�  A��
C��@E�Q�=���?
=qC�Ǯ                                    BxbF��  �          A&�R?�{�@���A�G�C���?�{�#\)?fff@�{C�(�                                    BxbF�z  �          A%G�@�R�Q�@���A���C��{@�R��H��\)����C�f                                    BxbF�   �          A$��@u���@H��A�  C��f@u�Q�}p���Q�C�                                      BxbF��  �          A#33@!G����@s33A�ffC��f@!G���þǮ�{C�33                                    BxbF�l  �          A (�@-p��  @X��A���C�c�@-p��G��=p���Q�C��f                                    BxbF�  
Z          A"=q@<(��  @8Q�A�=qC��3@<(������ff����C��                                    BxbF��  
�          A!G�@:=q�(�@)��At(�C��R@:=q�(��\�
�\C��H                                    BxbG^  T          A"=q@K��  @'�Ao�
C�}q@K���
���
�33C�E                                    BxbG  T          A$Q�@P  �(�@B�\A��C��@P  ��R��33��C�J=                                    BxbG*�  j          A#\)@mp����@<��A�Q�C�*=@mp��
=������(�C��)                                    BxbG9P  �          A��@W
=��
@3�
A��HC���@W
=�{���\���C�Y�                                    BxbGG�  �          A��@����G�@\)AZ�\C�L�@����Q쿯\)���C�                                    BxbGV�  �          A�@���@��AS�
C�� @����\������{C�]q                                    BxbGeB  T          A�R@�=q���@#�
A|z�C��@�=q���R�@  ���C�P�                                    BxbGs�  T          A{@�\)��ff@`��A�33C��@�\)��\>�@C�
C���                                    BxbG��  �          A��@����33@8Q�A�ffC��q@����(�����@  C��                                    BxbG�4  �          A�
?�\��?�G�@�ffC�K�?�\�{�4z�����C�}q                                    BxbG��  T          A�
?�=q�G�?�R@h��C��\?�=q�\)�\(���C�H                                    BxbG��  �          A  @���@�Aa�C�j=@��(��������C�Ff                                    BxbG�&  �          A��@HQ����H@�G�B{C��@HQ����@	��Ao33C�t{                                    BxbG��  �          A
=@A���z�@ÅB;  C��@A��߮@aG�A�z�C���                                    BxbG�r  �          A��@(���p  @�p�Ba��C��\@(����G�@��B�C�`                                     BxbG�  T          @�@0������@�=qB9�C���@0����{@K�AÅC���                                    BxbG��  "          @�Q�@Dz��\)@��Bu  C�Q�@Dz���@�{B/C�b�                                    BxbHd  
�          @�@s33�O\)@�Q�Bl��C���@s33�\(�@��HB=�HC��                                    BxbH
  "          @�
=@%����
@��BC�O\@%���Q�?�=qA\(�C�R                                    BxbH#�  �          @��
@Dz���@��B G�C�U�@Dz����@{A�ffC�#�                                    BxbH2V  �          @��H@3�
���
@p�A��HC��@3�
�ָR��R��33C�\)                                    BxbH@�  T          @陚@C33��G�@���B�HC���@C33�˅?���Ag
=C��3                                    BxbHO�  "          @���@@����ff@�z�B�HC�J=@@����ff@�\A���C��                                    BxbH^H  8          @�\)@L(����@Z�HA��
C�Z�@L(��ə�?h��@��HC�k�                                    BxbHl�  p          Ap�@��\��Q�@��HBDQ�C��@��\�tz�@��RB�C��=                                    BxbH{�  h          A33@�(���@�G�BL33C���@�(��y��@�33Bz�C�&f                                    BxbH�:  �          A��?��H��\?   @EC��?��H����S33��33C��                                    BxbH��  �          A{?���  >�G�@-p�C���?���	��X�����
C�B�                                    BxbH��  T          A?�ff����\)����C�  ?�ff�=q�u��ffC�c�                                    BxbH�,  �          Az�@����R?��
@ƸRC���@������3�
��{C��R                                    BxbH��  �          A{@����?�=q@�=qC��3@����&ff��ffC�33                                    BxbH�x  
�          A�?�����\@a�A���C��q?���������=qC�L�                                    BxbH�  
�          A{?��H�\)@x��A���C��R?��H�z�<#�
=uC��                                     BxbH��  
�          A��?����@L��A���C���?���p��B�\��ffC�Ff                                    BxbH�j  �          A   ?�Q���@\��A���C�1�?�Q��p��
=�X��C��H                                    BxbI  T          A!G�?����\)@q�A���C�AH?����33��\)����C��                                    BxbI�  �          A   ��p���
@��HA�{C��\��p���>���?��C��                                    BxbI+\  �          A#33>�p��G�@���A��C�#�>�p��"�R��\)�\C��                                    BxbI:  j          A$(�?����
=@uA�{C�=q?����#
=������C�                                      BxbIH�  p          A$Q�?���R@k�A�C��=?��!����(��=qC�,�                                    BxbIWN  
�          A$(�?�ff���@N�RA�
=C�XR?�ff� �׿c�
���C�
                                    BxbIe�  T          A"�H@	���Q�@a�A��RC��@	���ff�   �4z�C�q                                    BxbIt�  �          A$��@!G���@�=qB	��C��H@!G��z�@z�A9p�C�<)                                    BxbI�@  
�          A%�@%�z�@�
=A�Q�C�9�@%�
=?L��@�\)C�O\                                    BxbI��  
�          A$z�@&ff���@%�Aip�C��q@&ff�zῼ(���
C�p�                                    BxbI��  "          A&{@���@�
AN�HC��{@��\��ff� z�C���                                    BxbI�2  �          A%��@����@{A]p�C�ٚ@�����z���
C��H                                    BxbI��  �          A%?��
�p�@/\)AuC�j=?��
�!녿���z�C�K�                                    BxbI�~  >          A&{@!��G�@�
A733C�7
@!��p����4��C�5�                                    BxbI�$  �          A&{@3�
��@�RAFffC�
=@3�
����=q�"�\C���                                    BxbI��  �          A'�@hQ��p�?�
=A*�RC�]q@hQ�����\�4(�C�c�                                    BxbI�p  "          A'\)@O\)���?�=qA\)C�&f@O\)���(��X  C�N                                    BxbJ  T          A&=q@r�\�@
=AQ�C�f@r�\�z��ff�	G�C�ٚ                                    BxbJ�  T          A'�@g
=��H@)��Ak
=C�}q@g
=�33��ff����C�7
                                    BxbJ$b  T          A'�@aG����@�AQ�C��@aG��������z�C��{                                    BxbJ3  "          A&�R@r�\�z�?�{A�\C�ٚ@r�\�=q���K33C���                                    BxbJA�  �          A&�R@���ff@p�A[33C�n@���{�����{C�(�                                    BxbJPT  �          A&ff@���z�@>{A�G�C���@����
�8Q��~{C�aH                                    BxbJ^�  
�          A"�R@z����@��B�C�J=@z����?�Q�AJ{C�Ǯ                                    BxbJm�  T          A33@�
���@�BNG�C��@�
���@�=qA�\)C��)                                    BxbJ|F  T          A�\?�(��	��@���AڸRC���?�(���H?L��@�p�C��                                    BxbJ��  
�          A�@:�H��
=@��\B=qC�f@:�H��H@"�\Ar=qC��                                    BxbJ��  
Z          A=q?�  �p�@���A�33C��?�  ���?.{@��HC�L�                                    BxbJ�8  T          Aff@ ���G�@��AҸRC���@ ���G�?(��@|��C��                                    BxbJ��  �          A33@?\)��@_\)A���C��{@?\)��R��Q��C�f                                    BxbJń  "          A�@/\)�(�@s�
A���C���@/\)�G�>L��?���C�0�                                    BxbJ�*  �          Aff@ ����@y��A��RC�@ @ ����>��R?�C���                                    BxbJ��  �          A��@����?�z�AG\)C��R@��ff���H��
C���                                    BxbJ�v  
Z          A��?��
��?B�\@�C���?��
����*�H���C�33                                    BxbK   �          A=q@C33�	?�(�AC
=C���@C33�33��G��Q�C��=                                    BxbK�  �          A�R@����\)@c�
A�ffC�!H@����>L��?�C��                                    BxbKh  T          A\)@�z���@�=qAͮC�xR@�z��\)?�33@��C��f                                    BxbK,  
�          A!p�@����p�@�p�A�Q�C�k�@�����?�(�@߮C���                                    BxbK:�  �          A#�@�(��ff@c33A��\C�l�@�(���=L��>���C�t{                                    BxbKIZ  T          A&�\@�Q��=q@>{A��HC�t{@�Q�����(���h��C��H                                    BxbKX   �          A&�H@p���
=@�
AL��C��)@p�����p���\C���                                    BxbKf�  �          A(Q�@��R��\?�A(��C�\@��R��H����� (�C�
=                                    BxbKuL  �          A(��@~{���@
=A8(�C�J=@~{�{�ٙ��  C�4{                                    BxbK��  T          A)�@�G����@�G�A��HC��@�G��33?�@J�HC���                                    BxbK��  �          A(Q�@��
��@���B�RC�9�@��
�
=@;�A��C�<)                                    BxbK�>  �          A*�H@��H�
�H@eA�=qC�Ff@��H�=q<��
=�C�e                                    BxbK��  
�          A-��@�=q��@{�A�ffC��
@�=q���>u?��\C���                                    BxbK��  �          A.�R@�����@�A�z�C�` @����=q>��@{C�K�                                    BxbK�0  T          A0Q�@�����@H��A���C��@����\)�+��^{C���                                    BxbK��  "          A/33@�
=���@5�Ao\)C�y�@�
=�\)��  ��\)C�3                                    BxbK�|  �          A2�\@�����?���A!G�C��@�����33���C��                                    BxbK�"  T          A3\)@����?�G�@��C��@������ff�AG�C�K�                                    BxbL�  �          A3�@�Q��#
=?�z�@�RC���@�Q����!G��O33C�'�                                    BxbLn  �          A&ff@������?^�R@���C���@����
=�*�H�o33C�o\                                    BxbL%  "          A%p�@�Q���?ٙ�A�C��
@�Q��G���
=�,  C���                                    BxbL3�  �          A(��@������>�  ?�\)C�b�@����33�U���C��                                    BxbLB`  
�          A+33@�G���\�#�
�Z�HC��{@�G�����{����C��q                                    BxbLQ  �          A,Q�@\)��R��G���(�C��
@\)�G���(���{C�*=                                    BxbL_�  
�          A,z�@g
=�"�\�u���
C��@g
=�p���  ��{C��                                    BxbLnR  
�          A*�\@:=q�#�
�����
=qC��@:=q���p���Q�C��                                    BxbL|�  "          A&�\@5�\)�c�
��z�C��{@5��H��Q���C��3                                    BxbL��  �          A'�@ff�"ff��p�� ��C��H@ff������H����C�{                                    BxbL�D  T          A(Q�@��#33�5�w�C��3@�����p����C�b�                                    BxbL��  �          A)�@333�"�\���7�C��{@333�  ��\)���C�l�                                    BxbL��  �          A)�@J�H�!���Q��Q�C���@J�H����������C�|)                                    BxbL�6  
�          A,z�@^�R�!�B�\��=qC�� @^�R����p��¸RC�p�                                    BxbL��  
�          A-G�@h���"ff�n{��C���@h��������\��z�C��                                    BxbL�  �          A+�@=p��#�������=qC�\@=p����������C���                                    BxbL�(  �          A,z�@*=q�$  ��{�\)C�C�@*=q���
=����C�XR                                    BxbM �  �          A-G�@!G��'
=�c�
��
=C��@!G��ff��(��˅C�}q                                    BxbMt  
q          A/\)@?\)�(z������C��@?\)��\���R����C��q                                    BxbM  �          A0Q�@I���(�׾��
��z�C�P�@I���\)���
��{C��)                                    BxbM,�  �          A3\)@^�R�)녾�G��  C�\@^�R��
��Q�����C���                                    BxbM;f  
q          A3�@~�R�'33�:�H�n{C�k�@~�R�����R���C�aH                                    BxbMJ  
C          A333@��
�#\)�
=�AG�C�+�@��
��������p�C�1�                                    BxbMX�  �          A4��@����!�\��z�C�0�@�������\)���C�,�                                    BxbMgX  �          A/
=@�=q�{?�@-p�C�\)@�=q�
=�8���u��C��R                                    BxbMu�  �          A,Q�@���?��R@��
C���@���
=�
=�5��C���                                    BxbM��  
�          A.�H@�
=��׿k����C�C�@�
=������
����C�q�                                    BxbM�J  T          A4(�@ ��� (������+33C�
=@ ����  � (��{  C�xR                                    BxbM��  T          A,  ?��
�����
�Z�
C�,�?��
��R�$��k�C��                                     BxbM��  �          A*�H>��
�=q�$z�W
C��3>��
?�z��'\)¤L�B�                                    BxbM�<  �          A7\)@
=����\)��ffC�8R@
=����(��I�C�~�                                    BxbM��  
�          A2�R?�=q�\)���
�p�C�H�?�=q��(���\�oz�C�,�                                    BxbM܈  �          A2�H@qG����������
C��@qG���33���R�6{C�k�                                    BxbM�.  �          A7�?333�
{�����#��C�Q�?333����!���v\)C��\                                    BxbM��  
�          A8z�?�33�z������*
=C�w
?�33��  �#33�yffC�h�                                    BxbNz  �          A9p�@�\��33�=q�:Q�C�]q@�\����(z�u�C�c�                                    BxbN   �          A7
=?����ə���[�HC�.?����&ff�/��\C��                                    BxbN%�  �          A2�\?�  ������R�p��C�0�?�  ����.�\�)C�W
                                    BxbN4l  |          A2�H?�z��n�R�%G��C��)?�z�>��/�¢��@�z�                                    BxbNC  
�          A0  ?����=q�)��Q�C�z�?���?����,z�� B4(�                                    BxbNQ�  T          A,Q�?333����'33�\C���?333?����*ff£By�R                                    BxbN`^  
�          A+33?��H���&{�C�H?��H?�p��(z�ffBJ�
                                    BxbNo  @          A*ff?˅�[���HW
C�xR?˅>u�(  ¢�\A\)                                    BxbN}�  
�          A*{?�z��  �%p��{C���?�z�@��#�aHBC�                                    BxbN�P  
�          A(��?��þ\)�'�¥� C��?���@_\)���B�z�                                    BxbN��  
Z          A)�?��\�.{�(z�¦�C�4{?��\@^�R�
=#�B��
                                    BxbN��  �          A+�?�(�>�
=�)G�£�A��\?�(�@�������B���                                    BxbN�B  
�          A'�?\�l(��ff{C�9�?\�\)�%�£#�C�O\                                    BxbN��  �          A)G�?��R�c33��G�C���?��R=L���'�
¦�=@��                                    BxbNՎ  T          A&�R?�
=������\�w{C��\?�
=�����#�
 (�C��{                                    BxbN�4  �          A&=q?�z��hQ������C�� ?�z����#�¤.C��                                    BxbN��  �          A33>�{��  ��ff�?�C�~�>�{�j=q��
�\C��f                                    BxbO�  �          A$��>��������\�UC�7
>���:=q����C��                                    BxbO&  �          A'\)?#�
���\�
=q�_G�C��?#�
�#33�!��C�                                    BxbO�  T          A%��?s33�0  �Q�C�~�?s33?#�
�=q¦��B�                                    BxbO-r  
P          A'33�#�
�(���&=q¬C�k��#�
@9���   �3B��                                    BxbO<  @          A'
=?�p���=q�!��C�k�?�p�?��!��{BBQ�                                    BxbOJ�  �          A'
=@]p��.�R��\C��)@]p�?���z�z�Az�                                    BxbOYd  �          A"�H?��H�7��=q�)C�g�?��H>�����=An�R                                    BxbOh
  
�          A+���{@`  ���RBޏ\��{@�z����I�B�33                                    BxbOv�            A1���Q�@�,  p�B��Q�@����H�lB���                                    BxbO�V  
�          @��?p����\��ff�}
=C�Y�?p�׾�(����
�C��
                                    BxbO��  
�          A\)?�
=��
=���R�o(�C�P�?�
=������\�{C��{                                    BxbO��  �          A�?�����33��(��1  C�@ ?����Z=q���z�
C���                                    BxbO�H  
�          A2�R@J�H�G�@�{A��\C��@J�H�*�R?0��@a�C�C�                                    BxbO��  T          A5��@(��	p�@�  B{C�&f@(��%p�@aG�A�p�C���                                    BxbOΔ  
�          A1@=q�{@�
=B=qC��{@=q�&�H@;�AuG�C�~�                                    BxbO�:  
�          A7�
@)���=q@�p�B  C��@)���,  @B�\Av{C��=                                    BxbO��  T          A6�H@c�
�!�@w
=A�=qC���@c�
�-G�>��R?���C��                                    BxbO��  "          A6=q@@���p�@��B��C�'�@@���)@5�AfffC��                                    BxbP	,  �          A2ff@��ff@�p�B��C�@��#33@r�\A�33C�o\                                    BxbP�  �          A0��?�
=��@�B Q�C���?�
=�!��@���A�G�C�e                                    BxbP&x  
�          A1��?�{�
=@�B#
=C�h�?�{�!��@�p�A��HC�9�                                    BxbP5  �          A1p�?��H��@�p�B7\)C�u�?��H��@���A���C�C�                                    BxbPC�  �          A0(�@)����@�z�B*�C���@)���p�@���A�\)C���                                    BxbPRj  �          A.{@l����@��HB  C�q@l���(�@EA�  C�c�                                    BxbPa  �          A-p�@?\)���@�\)B=qC��R@?\)��R@L(�A�
=C�c�                                    BxbPo�  �          A,  @QG��33@���B	��C��q@QG��\)@EA�{C�K�                                    BxbP~\  T          A/�
@%����@���A�{C���@%��(��?���@�\C��)                                    BxbP�  �          A5G�@Dz���@���A�C�7
@Dz��)G�@ ��AMG�C�
                                    BxbP��  �          A-�@   �\)@ƸRB��C�>�@   �   @K�A���C�f                                    BxbP�N  
�          A.{@ ���p�@�{A��
C��
@ ���%��@  A>�RC�Ф                                    BxbP��  �          A)�@��Q�@�(�A��C���@�� z�@�
AIC��f                                    BxbPǚ  �          A (�@ff�z�?���A*�HC��\@ff�녿��
���HC�޸                                    BxbP�@  �          A#\)@���p���33��\)C��{@���\)���R��\)C�xR                                    BxbP��  �          AQ�?�33��׿���33C�+�?�33�	���H�ә�C��q                                    BxbP�  T          AG������33�}p���ffC��f�����
ff�z�H����C���                                    BxbQ2  �          Ap����\)�
=�_33C5������������
=C}{                                    BxbQ�  �          Ap����Q��  �(  C�'����  ���\��G�C�t{                                    BxbQ~  �          A�׾��H�\)�G��@z�C������H�����(���Q�C�O\                                    BxbQ.$  �          A�?��H��ÿ�{�/
=C���?��H��
��Q���(�C��                                    BxbQ<�  �          A33@ ���{����� z�C��@ ����
���R��Q�C���                                    BxbQKp  |          A*ff?&ff�ə����H�/�C��?&ff���H��  �w�\C��H                                    BxbQZ  �          A4  ��ff�
=�-��Q�Ces3��ff?�\)�/��C�                                    BxbQh�  T          A8  �(��p��0  �Cdh��(�?����3��=C�                                    BxbQwb  �          A9����G��C33�0��ǮCs����G�?��7\)¤�C!�                                    BxbQ�  �          A9p��G������5G��fCQ^��G�@�H�1�ffCٚ                                    BxbQ��  �          A2�H�˅���/�
¢�
CD��˅@:�H�)��.B�                                      BxbQ�T  �          A2=q>aG���G��0(�®�fC���>aG�@=p��)�B�\)                                    BxbQ��  T          A0�׿���1G��*ffB�C�\)���?�R�/�
«Q�C��                                    BxbQ��  �          A2�H����L(��)��C{�q���>W
=�1p�©�C(�3                                    BxbQ�F  �          A1�@����=q������C�aH@��������ָR��\C��
                                    BxbQ��  �          A2=q@^�R�����  �Q��C�3@^�R�+��z���C�*=                                    BxbQ�  T          A(  @*�H����R��RC�R@*�H��p��=q�MC�                                      BxbQ�8  �          A&ff@4z���=q�Ǯ��C��@4z��������Tp�C�q�                                    BxbR	�  �          A)�@0����H����C�@0���ə�� Q��E�\C�Ф                                    BxbR�  �          A(  @Q��z���=q��  C�Ф@Q��أ���
=�=��C���                                    BxbR'*  
�          A/�@ff�(�������33C���@ff��z��p��A=qC�y�                                    BxbR5�  �          A/�?�ff�&ff�6ff�s�C���?�ff�����H�{C��                                    BxbRDv  T          A3
=?Ǯ� ����p���C�g�?Ǯ�z���
=�"z�C�T{                                    BxbRS  �          A0��>�z��*�H���
��\C�Ǯ>�z���
������{C��)                                    BxbRa�  T          A5G�@��
����������{C��@��
��=q��z��3�\C�\)                                    BxbRph  �          AF�\@�ff��(���H�>{C��)@�ff���\�.{�s�HC���                                    BxbR  �          AF{@�=q�p�����C�H�@�=q�������KG�C���                                    BxbR��  
�          AD  @���p������z�C���@�������ff�((�C�Q�                                    BxbR�Z  �          AC33@�33�����=q�ٙ�C���@�33����\)�)�C�Y�                                    BxbR�   "          AD  @����"�R���
�ř�C�H@�������(�� (�C�o\                                    BxbR��  �          AF�\@�G��.�\������33C�(�@�G��\)��p���C��
                                    BxbR�L  �          AC
=@Z=q�5���2�\�S�
C�^�@Z=q� ������뙚C�c�                                    BxbR��  T          AD��@u�6=q�%��B{C�Q�@u�"=q��G����HC�`                                     BxbR�  �          AF�\@�z��6=q��G����
C�
@�z��(z��������C��f                                    BxbR�>  �          AFff@��
�1�=�G�?   C��@��
�*�R�G��jffC�\)                                    BxbS�  
�          AF�\@���5��?��
@��RC�(�@���3�
��
�C�C�                                    BxbS�  T          AJ�R@����4(�@H��Af=qC�~�@����:�H��  ��{C�q                                    BxbS 0  �          AL(�@���/�
@���A�\)C�Ff@���=p�?�Q�@��
C��=                                    BxbS.�  �          AN�R@��\�&ff@�  A�{C�K�@��\�<(�@@  AW33C��                                    BxbS=|  k          AO33@�  ��@�\)B�
C�Z�@�  �6{@�p�A�33C���                                    BxbSL"  =          AS33@���)G�@��A�\C�\@���?
=@AG�AT  C���                                    BxbSZ�  "          AS33@��R�)G�@�{A�(�C�n@��R�?�@I��A]C�7
                                    BxbSin  9          AO\)@|���-p�@���A؏\C��@|���A�@$z�A733C�\                                    BxbSx  �          ANff@������A��B0�HC���@�����H@�G�A�C��
                                    BxbS��  �          AK
=@������A\)B4(�C��@����=q@�  A�
=C���                                    BxbS�`  �          AK�@���� Q�A
{B+
=C�c�@����"{@���A�{C��                                     BxbS�  
�          AI�@��R��RA��B&�RC�O\@��R�"�H@��A��
C���                                    BxbS��  �          AI�@�  �   Az�B+�\C���@�  �!�@�{A�ffC��                                    BxbS�R  �          AH��@�
=�=q@�B��C�W
@�
=�*�R@�=qA��HC�]q                                    BxbS��  "          AF�H@�����Az�B;
=C�"�@�����\@�p�B(�C���                                    BxbSޞ  �          AE�@��\�z�@���B33C�q@��\�(z�@�G�A�Q�C�*=                                    BxbS�D  �          AG\)@�G��  @�
=BQ�C��f@�G��/�
@tz�A��\C�R                                    BxbS��  �          AG\)@�  �'33@�=qA�p�C�w
@�  �8(�@{A$��C��3                                    BxbT
�  �          AE�@�ff�	@�  B��C�Q�@�ff�%��@��\A�(�C�8R                                    BxbT6  �          AJ=q@�����H@�
=B�C�!H@����/�
@�33A�C�y�                                    BxbT'�  �          AIG�@�����
@�
=B��C���@����3�@r�\A��\C�&f                                    BxbT6�  �          AHz�@����p�A�B1�C�C�@�����@�A�{C��H                                    BxbTE(  �          AI�@�  ��\@���Bz�C��{@�  �(  @��\A�z�C���                                    BxbTS�  
�          AJ�R@�Q���@�p�A��\C��f@�Q��-G�@w
=A�=qC���                                    BxbTbt  �          AJ�\@�{�!G�@���A��
C�|)@�{�5p�@Dz�A`Q�C�=q                                    BxbTq  �          ABff@����=q@���A���C�
@����1G�@7�AZ�HC���                                    BxbT�  !          AF�\@����&�H@�  A�
=C���@����7\)@�RA&ffC���                                    BxbT�f  �          AE��@���&�R@�\)A��RC�g�@���3�
?�G�@���C��R                                    BxbT�  �          AK�@�ff�-�@�=qA�33C��f@�ff�8��?}p�@��C��                                    BxbT��  �          AC33@����+33@W
=A�(�C��@����3\)>�p�?�  C���                                    BxbT�X  
�          A@��@����%�@�
=A�C�=q@����1�?��@ə�C���                                    BxbT��  T          AB�H@�z��/�@3�
AV{C��@�z��5G��aG����\C���                                    BxbTפ  �          A:�H@X���(  @x��A�33C��\@X���2�\?s33@�p�C�p�                                    BxbT�J  
�          AG�@�����@��A�{C��)@����(��@)��AF�HC�c�                                    BxbT��  h          A?
=@�(�� ��@��HA�(�C�.@�(��/�@ ��A�C�Q�                                    BxbU�  �          A<��@U�-p�@AG�AnffC���@U�4  =L��>�  C�E                                    BxbU<  
�          A9�@\���,Q�@ffA((�C��H@\���/
=�O\)����C��                                     BxbU �  "          A<��@;��6�\>L��?s33C�1�@;��0���3�
�\z�C�h�                                    BxbU/�  "          A=G�@Mp��6{=u>�\)C��@Mp��0  �:�H�d��C�%                                    BxbU>.  "          A5G�@P  �,�þ�ff��\C�^�@P  �%��P����{C���                                    BxbUL�  �          A;\)@S33�-?���@��C�u�@S33�,�Ϳ�z��{C��H                                    BxbU[z  �          AF�R@�G��#
=@���A�C�g�@�G��3\)@(�A8  C�u�                                    BxbUj   T          AE�@e���@�p�B�C�Ф@e�*�R@�=qA�p�C�N                                    BxbUx�  T          AIp�@?\)����A Q�BR��C��@?\)��@��
B��C��
                                    BxbU�l  �          AH  @z��33AG�B;G�C��@z��$(�@�p�Bp�C�\)                                    BxbU�  T          A<��?(���
@�\B =qC��{?(��*{@�Q�AɮC���                                    BxbU��  �          A0zῐ���G�@��
B
=C�uÿ����%��@eA�=qC���                                    BxbU�^  T          A.{��ff�%�@C�
A��C���ff�,��>��R?�z�C��\                                    BxbU�  "          AI�@�Q����AQ�BE
=C��H@�Q����@�z�B  C��q                                    BxbUЪ  �          AJ=q@u���A
=BA33C��=@u��=q@�\)B�C��{                                    BxbU�P  
�          AH��@o\)�z�A
ffB.ffC�'�@o\)�#33@���A�{C��                                    BxbU��  T          AG�@e��p�A z�B ffC��@e��)G�@�G�A���C�U�                                    BxbU��  �          AH��@�33��
@�Q�B��C��R@�33�,Q�@�ffA�(�C��)                                    BxbVB  �          AF�R@E��=qAp�B"�RC��\@E��*=q@�33A֏\C�{                                    BxbV�  �          AD(�@Q��A33B)z�C��{@Q��"ff@��HA�C��{                                    BxbV(�  
�          AEp�@.{��\@�{B=qC�B�@.{�,��@��A��HC��                                    BxbV74  �          AE��@fff��H@�B  C��{@fff�,  @��A��C�C�                                    BxbVE�  �          AAp�@2�\��
@�{A���C��\@2�\�3�@aG�A�p�C���                                    BxbVT�  �          A?�@L(��#�@���A��HC���@L(��3\)@'
=AI�C��\                                    BxbVc&  :          A4  @z��%@j�HA��\C�Q�@z��/
=?�  @��C��q                                    BxbVq�  �          A0(�@��"=q@dz�A�p�C�|)@��+33?u@���C�&f                                    BxbV�r  �          AAG�@�{�
=@׮B  C��@�{�!G�@���A�\)C�                                    BxbV�  �          ABff@�{��@�
=A뙚C��\@�{�&ff@b�\A�\)C�AH                                    BxbV��  T          A8��@�\)�Q�@^{A�=qC���@�\)�%�?z�H@�Q�C�!H                                    BxbV�d  �          A/�@>�R�'33?�33@�=qC��{@>�R�'���Q���\)C��\                                    BxbV�
  �          A.=q?��
�)?�p�@��\C��f?��
�*ff�������C��H                                    BxbVɰ  
�          A0Q�?�\)�-�?z�@@  C��?�\)�*{��
�,��C��q                                    BxbV�V  �          A/�@	���*=q?��@�Q�C��3@	���)�����H��C��R                                    BxbV��  �          A/
=@B�\���@B�\A�=qC���@B�\�#�?��@Mp�C�H�                                    BxbV��  "          A,z�@
=��@��A\C�H@
=�$  @ ��A-�C�}q                                    BxbWH  �          A/
=@�p��
=@^{A��\C��@�p��   ?�{@�33C�S3                                    BxbW�  �          A/33@\(��z�?c�
@�G�C��@\(�����z���{C��q                                    BxbW!�  �          A333@���)��%�U�C�]q@���33��G���(�C��3                                    BxbW0:  �          A2�R@
�H�,  ��Q�� z�C��3@
�H��
�����C��                                    BxbW>�  �          A4��@,���+�����/�C�@,����\���
���C���                                    BxbWM�  "          A2�R?�  �(  �R�\��G�C���?�  ��H��p����RC�E                                    BxbW\,  �          A6{@ ���,Q��=p��r{C�G�@ ���Q�������
C��\                                    BxbWj�            A6�H?�
=�,���Y����G�C�!H?�
=�
=���H��
=C�z�                                    BxbWyx  �          A7�?z��*�R��(����C���?z���\��Q��
�C�                                    BxbW�  T          A6�\��Q��$����33�ɅC�  ��Q��ff���
���C��R                                    BxbW��  
�          A6ff�J=q�$Q����\����C�˅�J=q�����H�
=C�s3                                    BxbW�j  �          A6=q���H� (����
��\)C������H�  ��G��&33C�Y�                                    BxbW�  
�          A2{?+��&�H�u���C��
?+��(������{C��                                    BxbW¶  �          A8  �������33� �
CaH�����
=����4
=C|h�                                    BxbW�\  "          A8�׿�(��ff��ff��(�C���(�� Q�� (��1=qC��                                    BxbW�  �          A8�׽�G��*{�����\)C��{��G���������C��=                                    BxbW�  �          A8zῡG�����{��C�LͿ�G�����   �1(�C��3                                    BxbW�N  �          A8���"�\����33�p�C~u��"�\��(��  �=C{�                                    BxbX�  T          A9��������׮�p�C5������\�
{�@�
C{ٚ                                    BxbX�  �          A:{�*=q��H����!�\C|��*=q��Q����R�HCw��                                    BxbX)@  �          A9G��2�\��R��ff�&��C{#��2�\��
=����W��Cu�{                                    BxbX7�  T          A7�
�!��p������!G�C}#��!��θR�G��R�Cx�H                                    BxbXF�  :          A5�@���\�����ȣ�C���@��{�Y������C�>�                                    BxbXU2  �          A9p�>�{�"ff��  ��C��{>�{�
�\��(��%z�C��                                    BxbXc�  �          A8��>�G��$  ��=q��\)C�<)>�G������\)�!33C�p�                                    BxbXr~  T          A8  >����$Q���(���{C��=>����������p�C�                                    BxbX�$  �          A9p�����*�R��\)��  C��׾����\��\)�=qC��q                                    BxbX��  �          A5��>�  �&=q��  ��
=C���>�  �=q�����C���                                    BxbX�p  �          A4��?\)��
������C��q?\)�	p�����!p�C��                                     BxbX�  �          A4  >L���!��������33C���>L���Q���  �p�C��f                                    BxbX��  
�          A4��>\)�33���\��Q�C�e>\)�	����z���
C�t{                                    BxbX�b  "          A.=q���p���ff�  C�Ff���������>ffC��                                    BxbX�  "          A$Q�>��
��H�-p����RC�
>��
��z������\)C�33                                    BxbX�  T          A'33@��
���@��A�\)C�:�@��
���@E�A�33C��)                                    BxbX�T  
�          A+\)@���
=@_\)A�ffC��=@���  ?У�AQ�C�f                                    BxbY�  T          A,��@����@!�AX��C��@���
�R?+�@c�
C�<)                                    BxbY�  T          A"�R@�p��33?�@��
C���@�p��(���R�_\)C��\                                    BxbY"F  �          A!�@�p��(�@5A��C���@�p���R?p��@��C�*=                                    BxbY0�  �          A*{?����
=�����C�xR?���z��$��\)C���                                    BxbY?�  T          A.{?����R�\�%��C��?��ÿ�G��,z�¤p�C�`                                     BxbYN8  h          A*�R?�(��b�\�33�RC�z�?�(�����'� \)C�R                                    BxbY\�  �          A&�H?����(��
=Q�C��)?��=��
�=q¡�
@<(�                                    BxbYk�  T          A((���
=��{�
�\�kC{=q��
=�?\)�(�Q�Cp��                                    BxbYz*  "          A2�H@H����p��ff�h��C�B�@H���4z��#�#�C�                                      BxbY��  �          A.�H�aG���Q��
{�S�RC�f�aG��������(�C���                                    BxbY�v  �          A.�\?������
�  �N(�C��f?�����{�33�}Q�C��                                     BxbY�  T          A2�R?����33���N{C�ff?����(����}��C�"�                                    BxbY��  
�          A2�\>���33���BC���>���{���r�C���                                    BxbY�h  �          A2�H�\��\����+33C��;\�����p��[33C�O\                                    BxbY�  T          A1녿���\)��  ��\C�>������p����?\)C��                                    BxbY�  :          A/�?����"{�}p���C��?����p���
=�\)C�u�                                    BxbY�Z  <          A/\)���\)���H�ҸRC��\���  �׮���C�3                                    BxbY�   
�          A0Q��33�
=������  C�f�33�  ��p��Q�C�8R                                    BxbZ�  �          A,Q��������=q��Q�C�0������p���p��p�C�l�                                    BxbZL  �          A.{���\)��Q�����C��\����
���H�ffC��                                    BxbZ)�  T          A(Q�� ��������R���C~��� ����
������
C}�                                    BxbZ8�  T          A.ff�G���Q����9�C�
=�G��������hG�C�#�                                    BxbZG>  �          A,�;�����  ����R��C���������(��33�)C�(�                                    BxbZU�  T          A)���#�
��  ����@\)C��
�#�
���H���o
=C��{                                    BxbZd�  T          A'�@��
�  ��Q���RC�N@��
�(��z��8z�C��3                                    BxbZs0  �          A&�H@|(��\)�I������C��)@|(��=q��z�����C��f                                    BxbZ��  T          A(  @����	������C�1�@�������  ��C��f                                    BxbZ�|  l          A,z�@g
=�\)��(���=qC�=q@g
=��Q��أ��G�C���                                    BxbZ�"  �          A-�>�=q���H��p��@��C�R>�=q���
��R�n�C�s3                                    BxbZ��  T          A)녿�����G���m=qC��R�����W���ǮC{z�                                    BxbZ�n  �          A,  ��Q��s33�ffk�Cv��Q��G��'\)�RCbG�                                    BxbZ�  �          A'������\��
�y��C�J=���:=q� (���C���                                    BxbZٺ  �          A'
=@�p����H��G��"G�C��=@�p����
���R�F�\C�b�                                    BxbZ�`  �          A'�@xQ���\)��
=�.��C�w
@xQ���{����T=qC�R                                    BxbZ�  �          A'�
@Fff��=q��\)�>�C��@Fff��{���f
=C���                                    Bxb[�  �          A)G�?���Tz��ffǮC���?�녿�\)�!�  C���                                    Bxb[R  T          A)?�
=�����H�|C�.?�
=�,(��"=q�3C��)                                    Bxb["�  �          A)��?У���=q��
�h�C��f?У��^{���k�C��{                                    Bxb[1�  �          A)�@�
������oC�w
@�
�@  �{��C�4{                                    Bxb[@D  
�          A(��@p��\)���~�RC�}q@p��Q���B�C�H                                    Bxb[N�  
          A(  @:�H��z�����8��C��H@:�H����
=�`Q�C�H                                    Bxb[]�  "          A)�@=p���Q�����6��C�Ф@=p������^z�C���                                    Bxb[l6  �          A'�
@-p���\��R�.G�C�}q@-p���=q��H�V�
C�                                      Bxb[z�  T          A'�@K���33����\)C���@K����R��33�E�C��                                    Bxb[��  �          A'\)@���{�o\)��(�C�
@�������
=��\)C���                                    Bxb[�(  T          A(��@��
���a�����C�l�@��
��  ������p�C��\                                    Bxb[��  
�          A)��@��H�	���k���C��\@��H��  ��  ����C�                                      Bxb[�t  T          A*{@`  ��
=��{�	=qC���@`  �ָR���H�0�\C�Ǯ                                    Bxb[�  l          A*ff@B�\��33��ff�2��C�\)@B�\���
����Y�C�T{                                    Bxb[��  
          A&�R@u����9����C�E@u��(����\��G�C��                                    Bxb[�f  
�          A'�
@�33�
�H�|������C���@�33��G���Q�� {C��H                                    Bxb[�  �          A*ff@���z���33��C��@����{�����C��R                                    Bxb[��  �          A+\)@o\)� ����{���C�xR@o\)��33���H�(��C�T{                                    Bxb\X  �          A,��@�G���z���p��  C��=@�G���=q���5z�C�h�                                    Bxb\�  T          A-�@�����33���
�C��H@������H��(��.�\C��                                    Bxb\*�  �          A-G�@u���G���p��  C�R@u���
=��  �7z�C�Q�                                    Bxb\9J  T          A-�@���
=��=q�#Q�C�n@������
=�D��C���                                    Bxb\G�  �          A*�R����?��
AffC�Q���������k�C�XR                                    Bxb\V�  T          A&ff�.�R���@l(�A��HC}�)�.�R���?�
=A+\)C~n                                    Bxb\e<  �          A-p��`  �#
=?^�R@�33C{��`  �"�H��G����C{�                                    Bxb\s�  �          A*�\�?\)�#\)>aG�?���C}���?\)�!G������C}}q                                    Bxb\��  
�          A(z��s33�����R�\��Cxc��s33�G��
=�R�RCw��                                    Bxb\�.  �          A,(��^{����\��33Cx���^{���H�˅�(�Cv)                                    Bxb\��  �          A,Q��.{�=q������  C}޸�.{�
=������
C|(�                                    Bxb\�z  
�          A*�\�\)�33�U��Q�C��H�\)�
=�������C�3                                    Bxb\�   T          A*=q��\)��\��Q���C�����\)������p��9=qC���                                    Bxb\��  "          A)p�>��
��R�ə��p�C�q>��
�������?�RC�Q�                                    Bxb\�l  �          A*�R?����{���
�  C��
?����������<C��                                    Bxb\�  �          A+\)>�{��������gG�C���>�{�w���\#�C��                                    Bxb\��  T          A*=q�������H�%\CBB�����?�{�$��C!H                                    Bxb]^  T          A+�?\)�˅�(��¡�)C��?\)>��R�*�H­.A�                                    Bxb]  
�          A+
=��G���ff�(z�£.C����G�>����*=q°33B��                                    Bxb]#�  �          A)��=�Q��Y����Q�C���=�Q�˅�&�R¢��C���                                    Bxb]2P  �          A(z��G��G
=�33��C��{��G�����%p�¤��C{Y�                                    Bxb]@�  �          A'33@#�
���R��{�S{C���@#�
�\)�z��v��C�U�                                    Bxb]O�  T          A�\@�{����"�\�yG�C�Q�@�{��  �q���z�C�k�                                    Bxb]^B  �          AG�@G
=��
�tz���33C�@G
=��{����(�C�                                      Bxb]l�  �          A��@Vff�  �~{��Q�C��@Vff����33�{C�q                                    Bxb]{�  �          A33@��R��
=�aG����C��@��R�߮�������HC�Z�                                    Bxb]�4            A�@��R���H�!G��q�C���@��R��G��r�\���\C���                                    Bxb]��  n          A (�@~�R���R�����C���@~�R��G���{�=qC�.                                    Bxb]��  S          A�\@`����n{����C�aH@`�����H�����p�C�e                                    Bxb]�&  �          A"ff@��R�p�������C�l�@��R��  ��Q���C��=                                    Bxb]��  
�          A"�H@�z��z��A����RC�H�@�z���(���ff�υC�@                                     Bxb]�r  �          A#
=@�������������C��f@�������&ff�nffC��                                    Bxb]�  	          A&�R@��H��
�K����HC���@��H����(����C���                                    Bxb]�  	          A$(�@������<��
=�G�C�\@����{��
=��p�C�C�                                    Bxb]�d  
�          A$Q�@��H���
>.{?xQ�C��=@��H�ٙ���  ���HC�\                                    Bxb^
  
(          A$��@��
���
?#�
@eC��
@��
��(����H�,��C��\                                    Bxb^�  
�          A(  A�H��(�?O\)@�
=C�y�A�H��{��G��z�C�U�                                    Bxb^+V  
s          A-p�A{�p��?0��@n{C��RA{�tz�=#�
>uC�p�                                    Bxb^9�  
�          A+33A�R�w�@%�Ah��C���A�R��  ?��A)C��{                                    Bxb^H�  m          A&�\@�\��>��?O\)C�q�@�\���ÿ������C���                                    Bxb^WH  Y          A#�@���G��J=q���RC���@���	�����MG�C�)                                    Bxb^e�  	�          A#�@%��ff�1G��}��C���@%������������C�(�                                    Bxb^t�  
�          A/�Aff��(�@'
=A_\)C�s3Aff��
=?�(�@��C��q                                    Bxb^�:  Y          A+�@�\)��?   @,(�C�^�@�\)���ÿB�\���C�j=                                    Bxb^��  
A          A*�H@�\��  @�HAS\)C��H@�\���?�p�@���C�                                    Bxb^��            A/33A�����@S33A�C�h�A�����\@�AJ{C�P�                                    Bxb^�,  "          A0(�Ap��k�@n{A�p�C��Ap����@Dz�A�ffC�{                                    Bxb^��  
�          A/33Aff�p��@�p�A�z�C��{Aff��{@o\)A�p�C��                                    Bxb^�x  
�          A.�HAQ��:=q@���A�=qC�K�AQ��i��@�  A���C�=q                                    Bxb^�  T          A/33A  �6ff@��AǙ�C���A  �dz�@��\A�\)C��3                                    Bxb^��  
�          A.�RAp��\)@���AۮC��=Ap��QG�@���A�p�C�U�                                    Bxb^�j  "          A(��A33���R@#�
Al(�C��=A33����?�A��C��                                    Bxb_  
�          A)�A  ��ff?��@��RC��qA  ��G�<�>#�
C�˅                                    Bxb_�  '          A�@�Q��
�H���B�\C���@�Q���Ϳ�����C��=                                    Bxb_$\  
�          A\)@����p�?��Ap�C�E@�����H?   @9��C��=                                    Bxb_3  �          A!@�\)��z�@@  A�=qC�q@�\)����?�A$��C�\)                                    Bxb_A�  
�          A%�A33�S�
@��A���C�Q�A33�z�H@^�RA���C���                                    Bxb_PN  �          A z�@Å����@��HA¸RC�H�@Å��
=@=p�A�
=C�q                                    Bxb_^�  �          A&�H@˅����@�(�B=qC���@˅����@���B��C��)                                    Bxb_m�  
�          A,  @�����Q�@��A���C�Ф@�����Q�@�(�A��
C�>�                                    Bxb_|@  
�          A&ff@�����@��A�=qC�h�@���33@HQ�A�Q�C��                                    Bxb_��  �          A((�@�G���(�@��
Aܣ�C�` @�G���\@r�\A���C��f                                    Bxb_��  �          A%�@�z���ff@���A���C��@�z��Ӆ@i��A�ffC���                                    Bxb_�2  �          A#�@���33@�{AîC�>�@���{@R�\A��C��=                                    Bxb_��  "          A��@�Q����?��
A{C�Z�@�Q����?#�
@s33C��\                                    Bxb_�~  
�          A{@�������?��Az�C���@�����ff>��H@AG�C�33                                    Bxb_�$  �          A(�@�z���p�?s33@\C��@�z��Ǯ�#�
��  C��R                                    Bxb_��  "          A{@��
���
=�U�C���@��
�У��K���(�C�e                                    Bxb_�p  T          A{@u��{�-p����
C�E@u��p��xQ���{C�
                                    Bxb`   "          Ap�@l���׮������ffC�aH@l�����R�����ffC��f                                    Bxb`�  �          A�
?�����p���{�B��C��H?����������e  C���                                    Bxb`b  �          AQ�?�������(��R33C���?�������33�u�HC��                                    Bxb`,  T          A��>\���
�   �[p�C���>\�����z��33C���                                    Bxb`:�  �          A
=?�����=q���
�T(�C�W
?��������
�H�w(�C�Ф                                    Bxb`IT  T          Aff?aG��z�H�����C�N?aG��   ���33C��                                    Bxb`W�  �          AQ�=�������
�Ru�C�,�=��Ϳ����33¦B�C���                                    Bxb`f�  �          Ap��\)�&ff�33�C�p��\)��(��  ¤k�C���                                    Bxb`uF  �          A�ÿ�=q���ff£(�CI+���=q?z�H� z�C��                                    Bxb`��  "          A33�!�@@���\)u�C��!�@���ff�i�
B�Ǯ                                    Bxb`��  �          A�H��
@B�\��\��B�� ��
@���	p��o��B�G�                                    Bxb`�8  T          A��(�@@����\8RB�\�(�@������H�j=qB�\)                                    Bxb`��  �          A��@h������=q�ָRC�o\@h����
=���
�
z�C���                                    Bxb`��  T          A��@�����\)��\)��{C�T{@������������C�=q                                    Bxb`�*  
�          A��@�\)��׿���)��C�h�@�\)���1G�����C���                                    Bxb`��  �          A\)@�z��
{�=p���Q�C��@�z���\��\�=C�{                                    Bxb`�v  T          AQ�@�����׾�  ��  C��H@����ff��G��\)C���                                    Bxb`�  "          A33@�\)��  ?��H@�ffC��@�\)��33=�Q�?��C��R                                    Bxba�  �          A\)@�z����@.{A�Q�C�Z�@�z���  ?��HA6�RC���                                    Bxbah            A��@ۅ��
=@z�AP��C�3@ۅ��
=?�
=@���C�y�                                    Bxba%  
�          A{@�{�	�?�(�@�C�XR@�{�
�\��Q��C�9�                                    Bxba3�  
s          Ap�@��H�  ?�\A&ffC��f@��H��R>�(�@!�C�P�                                    BxbaBZ  �          A\)@�����?�
=@ڏ\C�O\@���=q���
���C�.                                    BxbaQ   T          A(�@�{���>u?���C���@�{���
�aG�����C��
                                    Bxba_�  T          A@�(��{@1�A���C�z�@�(��\)?��
A=qC��q                                    BxbanL  �          Az�@����  @5�A�p�C�aH@���p�?��A�C�ٚ                                    Bxba|�  T          A33@����?�\A+33C��@����G�?�R@l��C��
                                    Bxba��  �          A�@C33�ff@U�A�\)C��
@C33���@�\AE�C���                                    Bxba�>  T          A(�@B�\�p�@|(�A�  C�P�@B�\�	G�@,(�A���C���                                    Bxba��  T          A�@ �����@�A�C��@ ����R@^{A�ffC�L�                                    Bxba��  �          A=q?���
=@��
A���C�xR?���\)@7
=A�=qC�R                                    Bxba�0  �          A=q?�����R@s33A�{C��\?����=q@ ��AyG�C�Q�                                    Bxba��  �          A=q>���
�R@`��A�(�C�XR>���p�@(�AXQ�C�G�                                    Bxba�|  	          A�
@�����@���B ��C��@�����(�@�  AЏ\C�b�                                    Bxba�"  �          A(�@���љ�@�B��C�4{@����R@z=qA�33C���                                    Bxbb �  �          A�
@�
=���H@�(�A�33C�
@�
=��ff@fffA�Q�C��                                    Bxbbn  �          A�@�ff��@|��A�G�C�+�@�ff��@6ffA�Q�C�W
                                    Bxbb  T          A��@�����@�z�BG�C��)@������
@�p�A�  C�k�                                    Bxbb,�  �          A33@�{����@��A�G�C�b�@�{���@A�A��RC�}q                                    Bxbb;`  �          A=q@�Q��ƸR@_\)A�ffC���@�Q�����@"�\A�(�C�˅                                    BxbbJ  T          A��@�\)�ٙ�@5A�=qC��3@�\)����?���AEG�C�S3                                    BxbbX�  T          A�@�(���=q@p��A�
=C��@�(����@9��A���C�p�                                    BxbbgR  T          AG�@��
��
=@u�A��C�u�@��
��\)@C�
A��C�
                                    Bxbbu�  T          A�
@������
@i��A��
C��@������H@7
=A�\)C��q                                    Bxbb��  
�          A  @�����H@k�A�{C�q@���ٙ�@.{A��RC�!H                                    Bxbb�D  �          A\)@�z���{@�z�B\)C���@�z���p�@��\A�z�C��                                    Bxbb��  T          Aff@�����ff@��B*G�C���@�������@���Bp�C��                                    Bxbb��  
�          A��@�
=�g�@���BH�C���@�
=����@��B1Q�C�k�                                    Bxbb�6  �          A
=@��H����@�33BD��C���@��H�8��@�  B7=qC���                                    Bxbb��  �          A�H@�(��S�
@�ffBB��C��\@�(���  @���B.G�C�~�                                    Bxbb܂  �          A�@�p��e@�33B�HC�g�@�p����@���B	\)C��{                                    Bxbb�(  �          A��@ȣ��P��@�33B��C�Ff@ȣ�����@�=qB�C��H                                    Bxbb��  �          A�
@��H�_\)@���B��C�3@��H��  @��RB  C���                                    Bxbct  
�          A  @��H�,(�@��
B�C���@��H�W�@�A�G�C�L�                                    Bxbc  �          Ap�@ȣ׾�\)@p��A�p�C��3@ȣ׿Q�@k�A�C�B�                                    Bxbc%�  T          @�Q�@�33?�(�@AG�A��Aip�@�33?�G�@N�RAͮA-��                                    Bxbc4f  �          @�{@׮@�@7�A�p�A��
@׮?˅@HQ�A£�AT��                                    BxbcC  
�          @�33@�Q�@"�\@<(�A�G�A���@�Q�@ff@Q�AƸRA�                                    BxbcQ�  
Z          A Q�@���@j�H@ ��Aj{A�
=@���@U@!G�A�G�Aљ�                                    Bxbc`X  "          @�Q�@���@p  ?�p�A
=A�\)@���@a�?�G�ARffA��                                    Bxbcn�  "          @�(�@��R@�\)�ff��z�B.�R@��R@�\)��33�4��B4�                                    Bxbc}�  "          @�Q�@��
@�33��G���33B(@��
@�{��=q�33B*�
                                    Bxbc�J  "          @�ff@�(���@��A�
=C��{@�(���  @{A��C���                                    Bxbc��  �          @��@�(����H@:=qA��HC��@�(��
=@&ffA��C���                                    Bxbc��  T          @�Q�@ȣ��>{@-p�A��RC�W
@ȣ��Tz�@��A�G�C�
=                                    Bxbc�<  T          @���@~{��@(�A�\)C�R@~{��
=?��AT  C�b�                                    Bxbc��  
(          @ᙚ@������@5A�C��@����(�@Q�A�  C��=                                    BxbcՈ  
�          @�(�@�ff��G�@xQ�B�C���@�ff��ff@l(�Bp�C��                                    Bxbc�.  �          @��@��׼#�
@@  A��C��
@��׾�(�@>{AхC��                                    Bxbc��  �          @��@��H�
=?�A]p�C��@��H�%?��RA2=qC��3                                    Bxbdz  �          @�z�@��H�G
=?xQ�@��HC�]q@��H�N{?�@��C��q                                    Bxbd   �          @��@�G��|��?�  A��C��@�G����H?.{@�z�C�|)                                    Bxbd�  "          @��@�  ��p�?���A�RC�Ф@�  ����>�(�@S33C�}q                                    Bxbd-l  
�          @�p�@�ff���H?��@�=qC�0�@�ff���
�\)����C��                                    Bxbd<  "          @�=q@���,(�>�?�C��@���+���  �(�C��                                    BxbdJ�  T          @�{@�{@��5��ҸRAď\@�{@0  �p���  A�G�                                    BxbdY^  �          @�  @
=@c33��ff��33Ba�\@
=@qG����
���\Bg��                                    Bxbdh  �          @��?�G�@����У����B��\?�G�@���n{�$z�B���                                    Bxbdv�  
�          @�  ?��\@�p���(���  B��?��\@�zῘQ��9p�B��=                                    Bxbd�P  
�          @�����
@�Q�������RB�zὣ�
@�  ��p��+�
B�p�                                    Bxbd��  �          @�33��\@�
=�*�H���B�LͿ�\@�G���\�qB��)                                    Bxbd��  �          @��?�33��\)?(��A���C��H?�33��p�?(�A�(�C���                                    Bxbd�B  �          A@<(���@|(�AУ�C��
@<(���
=@5A�{C��                                    Bxbd��  
�          A=q@aG���@_\)A��C�S3@aG��p�@Q�Aq�C���                                    BxbdΎ  �          A��@7���\)@l(�A��
C��@7���R@!�A}��C�k�                                    Bxbd�4  
(          A��@j�H���H@~{A�C��=@j�H��@7�A�33C�<)                                    Bxbd��  �          A��@J�H���
@i��A�
=C�N@J�H� ��@"�\A�
=C��                                     Bxbd��  
�          A
=>�
=��\)@��A�33C��>�
=�Q�@N�RA���C�t{                                    Bxbe	&  
Z          A��Ǯ��@�A�{C��=�Ǯ��@Dz�A��C��H                                    Bxbe�  �          A
=�\��p�@��A�C����\��@HQ�A���C�H                                    Bxbe&r  �          A��
�H�ڏ\@g
=A�{C|^��
�H���@'�A�Q�C}^�                                    Bxbe5  
�          @�p��n�R���
@�=qB�HCk5��n�R��p�@aG�A��
Cm��                                    BxbeC�  
�          A ���o\)��(�@��
B��Ci�f�o\)��Q�@�33A��RCm�                                    BxbeRd  "          @�\)��p�����@�p�B+�C`33��p�����@���Bz�Cd޸                                    Bxbea
  
�          @����
=�QG�@�G�B8\)CY����
=��Q�@���B#G�C_�=                                    Bxbeo�  �          A�׾8Q���=q@��BK�RC���8Q���{@�  B+z�C�+�                                    Bxbe~V  	�          A�>.{���@���BI(�C��f>.{���@��HB(��C��f                                    Bxbe��  m          A	G�@z���p�@��HB+��C��H@z���@�ffB(�C���                                    Bxbe��  �          A�@#�
���@�G�A��C�P�@#�
���@C33A�{C���                                    Bxbe�H  �          A\)@Y����G�@L��A��
C��f@Y����p�@�Ao
=C�S3                                    Bxbe��  �          @�\?p����ff@�(�BC��f?p������@tz�A��C�l�                                    Bxbeǔ  �          @��
?�����\)@�ffB��C���?������@�B�\C�Y�                                    Bxbe�:  
�          @޸R@HQ�?�33@��
Bc�RA�p�@HQ�>�
=@�\)Bk�@�p�                                    Bxbe��  �          @��@8���(��@`  B'{C��=@8���G
=@EB\)C�xR                                    Bxbe�  �          @��\?˅@0���<(��*�Bp33?˅@I���!���B|�R                                    Bxbf,  �          @�z�@>�R?�z�@��Bd�
A��H@>�R?.{@�Q�Bop�AM�                                    Bxbf�  �          @�R@�
=@���@���B�
B;�@�
=@z=q@�z�B��B+=q                                    Bxbfx  
�          @�z�@���@�Q�?�33AmBG�@���@�{@ ��A��RB\)                                    Bxbf.  
�          @�@�@��?�33A�\A�33@�@p�?�p�A,��A��R                                    Bxbf<�  "          @�@�  @
�H?�z�A(�A�  @�  ?��R?���A4��A}                                    BxbfKj  �          @�Q�@�z�333=���?Y��C�{@�z�5    <�C��                                    BxbfZ  "          @���@�G�?=p�?��A=q@Ϯ@�G�?z�?�
=A#33@��                                    Bxbfh�  "          @��@��@�=q?�p�A��B4\)@��@�  @(��A��HB,\)                                    Bxbfw\  T          @�33@ȣ׿c�
@eA�=qC��@ȣ׿��@Z�HA��C��H                                    Bxbf�  �          @�G�@���!G�@���B7�C��@���QG�@�z�B'�C���                                    Bxbf��  T          @��H@���G
=@�=qB�HC�~�@���l(�@uA�Q�C�XR                                    Bxbf�N  "          @��@����1�@VffA�p�C���@����N{@;�A��C�+�                                    Bxbf��  �          @�G�@�(���\)?�(�AzffC���@�(����R?�{A+\)C�Ǯ                                    Bxbf��  T          @�p�@�z����\?}p�A   C��\@�z���p�>�=q@	��C��\                                    Bxbf�@  �          @�p�@�p����?��HA��C���@�p����
?�@�G�C�\)                                    Bxbf��  T          @��@�33�S33@9��A���C��q@�33�j�H@=qA�ffC�Z�                                    Bxbf�  �          @���@�\)�E�@��\B)�\C��{@�\)��Q�@�p�B"�HC�P�                                    Bxbf�2  "          @�@p��@��\@q�B	
=B=Q�@p��@`��@��B   B,                                      Bxbg	�  
�          @�(�@��
@N�R@{A�G�A�
=@��
@8Q�@)��A�=qA�\)                                    Bxbg~  
�          @�\@�@��?�Q�A{B5z�@�@���?��AuG�B0\)                                    Bxbg'$  �          @��@p��@��@G�A�
=BdG�@p��@�@C�
A��HB](�                                    Bxbg5�  
(          @��@�(�@��@�
A�z�BT33@�(�@��@C�
AǮBL=q                                    BxbgDp  	`          @�@�\)��ff?���Ag�C��)@�\)���?�z�AR{C��                                     BxbgS  �          @�  @У׿0��?��A��RC��{@У׿s33?�\Aq�C��{                                    Bxbga�  �          @��?:�H@�33?�  A\  B�ff?:�H@��\@ffA�z�B�                                    Bxbgpb  T          A��@���@Å@Z=qA��
Bbp�@���@�=q@��A�\)BXz�                                    Bxbg  �          A33@��H@j�H@�p�Bp�B �@��H@>{@���B�A׮                                    Bxbg��  �          AG�@��@�@��B�HA�G�@��?�=q@���B&�HAK33                                    Bxbg�T  
�          A��@�33>���@���B#  @q�@�33��33@���B#(�C�`                                     Bxbg��  �          A Q�@����G�@�Q�B  C�S3@�����H@�=qB��C�:�                                    Bxbg��  T          @�z�@�{�	��@�(�B��C�q�@�{�/\)@���A�  C��                                    Bxbg�F  �          @���@���I��@}p�A���C��
@���k�@^{A�  C�
=                                    Bxbg��  T          @�{@�  ��z�@#�
A�=qC�@�  ���R?�z�Ar�HC�\                                    Bxbg�  �          @��@�ff��G�?�A�Q�C��@�ff��Q�?�
=A33C�L�                                    Bxbg�8  
�          A Q�@�����@G�Aj{C��@�����H?�p�AffC�k�                                    Bxbh�  �          A ��@�ff��
=@*�HA�C���@�ff��G�?�{AW\)C���                                    Bxbh�  �          Az�@�=q���
@b�\Aģ�C���@�=q���@,��A�=qC���                                    Bxbh *  �          A ��@�����ff@:=qA���C�Z�@������@��A|z�C�e                                    Bxbh.�  "          A	�@tz�����@��\A��
C�j=@tz���p�@HQ�A���C�s3                                    Bxbh=v  �          A  @w
=�Ӆ@C33A��C�!H@w
=��\)@�Ai��C�w
                                    BxbhL  �          A�\@Dz���
=@5A�33C�� @Dz���?��
AD��C��                                    BxbhZ�  �          Aff@<(���{?�(�A!C�y�@<(���=q>�33@�C�P�                                    Bxbhih  �          A�@���޸R@`  A�\)C�� @����z�@{A���C��)                                    Bxbhx  �          A�
@   ���@��RB/��C���@   ���
@�(�BC�q                                    Bxbh��  
�          A�?�\����@޸RB^��C���?�\��33@�  B?��C�*=                                    Bxbh�Z  T          A	G�?�\)���R@�p�Be{C���?�\)��{@ϮBFp�C��f                                    Bxbh�   �          A
�\?}p��,(�A�\B�W
C��?}p��u�@�{B{{C�9�                                    Bxbh��  "          A  >�녿���A��B�.C�|)>���$z�A(�B�
=C���                                    Bxbh�L  "          Azᾅ��W
=@��B�G�C��{�����ff@�Bi�C�\)                                    Bxbh��  
�          A
=>�Q��mp�@�=qB�C�˅>�Q���Q�@޸RB^�C�.                                    Bxbhޘ  
�          @������XQ�@޸RB�(�C������33@���B_(�C�.                                    Bxbh�>  �          @��H?\(����@�33BO�\C��?\(����H@�ffB.�C�f                                    Bxbh��  "          @��?.{�\(�@�=qBQ�HC��)?.{��=q@���B1(�C���                                    Bxbi
�  	�          @�(�@33��(�@(�A���C���@33����?��
A�(�C�޸                                    Bxbi0  "          @���@z��\)@��A��
C��@z����?���A���C�%                                    Bxbi'�  
�          @�z�@-p���p�@���B�C��H@-p����R@XQ�A�33C��3                                    Bxbi6|  �          @��@33��  @��
B*�HC��f@33����@�(�Bp�C��H                                    BxbiE"  �          @�@���R@�33B���C��@����H@�ffB�aHC���                                    BxbiS�  
�          @���@���W�@�
=B6=qC���@���|��@k�B=qC��H                                    Bxbibn  
�          A�@�=q<#�
@��Bn{>8Q�@�=q��  @��HBj{C�                                      Bxbiq  T          A�\@;����@�p�Bw�C�@;��^{@�\)B_�HC�R                                    Bxbi�  �          A��@=q�AG�@�By=qC�G�@=q��=q@��HB]z�C�S3                                    Bxbi�`  T          @�(�@��[�@���Bf�RC���@����
@�=qBJ(�C��\                                    Bxbi�  
�          @��R@p���  @�=qBL  C��@p����H@�(�B.33C��                                    Bxbi��  
�          @θR?�
=��  @�  B233C��f?�
=���
@u�B�C���                                    Bxbi�R  T          @��\?˅�Z�H@dz�B-��C�s3?˅�y��@A�B��C�                                    Bxbi��  �          A  @�
��@S33A\C�33@�
���H@  A�(�C��R                                    Bxbiמ  T          A�@8����  ��z���
=C�=q@8������\�~ffC��                                    Bxbi�D  T          A��@�����\)>�?Y��C��)@�������z�H���C��R                                    Bxbi��  T          A  @�  ��p��:�H��=qC��f@�  ��\)���
�>{C�                                      Bxbj�  �          A�@�G���  ��ff�$��C��@�G���{�'
=���C���                                    Bxbj6  
�          A  @�p�����\)���C�z�@�p���\�(�����C���                                    Bxbj �  �          A�@QG���p���(��\)C�<)@QG����'�����C���                                    Bxbj/�  
Z          A	�@H����ff�E����C��=@H��������
���C��H                                    Bxbj>(  "          A��@3�
��ff?��\@��HC�H@3�
���׾\)�uC���                                    BxbjL�  T          A�@'
=��R����0  C��)@'
=� Q�У��.�RC��                                    Bxbj[t  �          A��?�Q�����@�  A�RC���?�Q����\@G�A�33C�7
                                    Bxbjj  T          A�\�   ��G�@��HB{C�箿   ����@a�A˅C�{                                    Bxbjx�  �          A�>\��G�@i��A��HC�� >\��  @!G�A�  C�h�                                    Bxbj�f  
�          A��?����{@Z�HA�Q�C�w
?�����
@33A�z�C�(�                                    Bxbj�  �          @�\)@
=�߮@@��A�33C�h�@
=��?�Aa��C�H                                    Bxbj��  T          @�(�@j=q�ƸR?8Q�@��C�@ @j=q��  �aG��޸RC�,�                                    Bxbj�X  �          @�@QG����R�a���C���@QG����
����G�C�j=                                    Bxbj��  
�          @�p���(�����ۅ¨W
Cg녾�(�>��H�ۅ©�CW
                                    BxbjФ  �          AG����?�ff��=q��C�
���@)����Q��{{C�=                                    Bxbj�J  T          @�(���Q�?�ff�\�U{C \��Q�@�������FQ�C)                                    Bxbj��  �          @�{���@!G������S
=C����@\�����\�>\)C��                                    Bxbj��  �          Ap��mp�@:=q���
�V(�C���mp�@u����>��C��                                    Bxbk<  
Z          @�G�?L��?aG���z�¡�B>�?L��?��H��p�
=B�p�                                    Bxbk�  �          A\)@B�\��=q@|(�BG�C�e@B�\���
@Q�A�
=C���                                    Bxbk(�  �          A�
@q���\)@�B�\C�Q�@q���ff@���A�{C���                                    Bxbk7.  T          A
{@l�����H@��
B
=C�0�@l�����@��A�C��)                                    BxbkE�  �          AG�@u����@�33B8C�u�@u����@���BQ�C��                                    BxbkTz  
�          A�@>{�L(�@��
Bp�C�q�@>{���\@�G�BU33C�7
                                    Bxbkc   "          A�\@����G�@ڏ\B?�
C�C�@����G�@���B$�C�xR                                    Bxbkq�  T          A33@����ff@ᙚBAG�C���@�����@ǮB%�C���                                    Bxbk�l  �          A33@\)��{@陚BJ��C��)@\)��Q�@���B/33C��3                                    Bxbk�  
�          A��@w
=���@�Q�B=��C�h�@w
=��{@�(�B (�C���                                    Bxbk��  "          A�@�{��33@��HB0ffC�f@�{���@�BffC��=                                    Bxbk�^  �          A��@^�R��p�@��
B-
=C��)@^�R����@��B�C��                                    Bxbk�  
�          A	��@W���G�@�(�B=qC��
@W��أ�@�(�A�
=C�@                                     Bxbkɪ  T          A@XQ���G�@��A��C�!H@XQ���(�@Q�A�33C�3                                    Bxbk�P  T          A	��@�\)��G�@�An{C�S3@�\)����?��HAp�C���                                    Bxbk��  
�          A
�R@ƸR���R�����G�C��\@ƸR����R�qC�g�                                    Bxbk��  "          A
ff@�ff���R���\(�C��@�ff������Q���\C�q�                                    BxblB  
�          A�H@ָR��  �0����ffC��f@ָR���\��(�� ��C��                                    Bxbl�  
�          A�R@�����
������=qC��@����z��\�Dz�C���                                    Bxbl!�  �          A{@�z��{���p��@(�C��3@�z��g
=����{C��
                                    Bxbl04  �          A�
@����
=�h����=qC�N@����(����\�33C���                                    Bxbl>�  
�          A(�@�{��
=>�=q@�
C�{@�{��ff�\)��Q�C�%                                    BxblM�  �          AQ�@�p����R@   A��RC�� @�p�����?ǮA-�C��
                                    Bxbl\&  �          @�Q�@�
=��=q@e�A�=qC�1�@�
=��=q@+�A�C��{                                    Bxblj�  �          Az�?�\)@�R���BG�H?�\)@Y����=q�u{Bt�R                                    Bxblyr  
�          A��?�33?�G��(�G�Bb��?�33@Fff���qB�(�                                    Bxbl�  
@          A��?��?�=q�	����BE�?��@,(��z��B�.                                    Bxbl��  
�          A�\?Q�?��H��R��B��?Q�@HQ��z��B��q                                    Bxbl�d  T          Az�?�
=@���	��
=Bd=q?�
=@c�
�{�B�G�                                    Bxbl�
  �          A\)@ff@G���G��qB<@ff@_\)��=q�s��Bk��                                    Bxbl°  
�          A\)@�  �����  ��\C��q@�  >k������	G�@z�                                    Bxbl�V  
          A ��@θR�J�H�`����{C��@θR�#�
�~�R��Q�C�33                                    Bxbl��  
          @�  @��
�g
=�mp���RC�:�@��
�=p�����(�C��{                                    Bxbl�  �          @�Q�@�����
=@�ffB�C��q@�����33@Z�HA�  C�&f                                    Bxbl�H  �          @��@�Q���=q@,��A��RC�O\@�Q���ff?�A�=qC�*=                                    Bxbm�  
�          @���@�������@�A�C�!H@������\?��\A�C�\)                                    Bxbm�  �          @���@�=q���@Ayp�C�3@�=q��Q�?��RA\)C�P�                                    Bxbm):  "          @��@�=q��p�@
�HA���C���@�=q���R?�\)A"�HC���                                    Bxbm7�  "          A�@���q녾\)�xQ�C�W
@���l�ͿB�\��G�C���                                    BxbmF�  "          A�@����.�R��=q�0(�C�+�@����(�� ���`��C�+�                                    BxbmU,  �          A�H@��H��{�z�H���
C���@��H��
=�ٙ��1p�C�xR                                    Bxbmc�  �          @���@���\�dz����C��3@���z��s�
�	�C���                                    Bxbmrx  �          @�\@��\������Q�C�` @��\��ff��z��p�C��H                                    Bxbm�  "          @�33@�G��������H�?��C�%@�G�?(���=q�>�@��
                                    Bxbm��  
�          A\)@������N�R�£�C��@����e�x����\)C�#�                                    Bxbm�j  
�          AQ�@�ff�>�R�AG���C��=@�ff�(��^{�{C�J=                                    Bxbm�  
�          A
{@��H��p������4�C�
@��H�Fff��z��L�
C�=q                                    Bxbm��  
�          A	�@AG����\���\�=C���@AG�=u��G�?���                                    Bxbm�\  T          A�R@�  ���H��  �<��C�J=@�  ��{�=p���{C���                                    Bxbm�  �          A�
@u��33�XQ����C���@u��p���=q��C��R                                    Bxbm�  
�          A33@7�����w
=��  C�` @7��ٙ����\�
(�C�k�                                    Bxbm�N  �          A\)@U����
?   @VffC�w
@U����H�W
=��(�C��                                     Bxbn�  "          A
{@{���\)��G��  C���@{�������z��1C�k�                                    Bxbn�  �          A(�@��
��(���p��)�RC��@��
�Fff��G��A�
C��                                    Bxbn"@  
r          A
�\@��
�;���33�IffC���@��
��G���  �[�C���                                    Bxbn0�  "          Ap�@��
������
�F=qC�� @��
?+���33�E�@�=q                                    Bxbn?�  �          A(�@�=q?�{�����)�A��@�=q@Q�������A��                                    BxbnN2  "          A
=@�Q�.{���
�9��C��=@�Q�>�(���z��:@��\                                    Bxbn\�  �          @��R@У׿Y������p�C�Ff@У׽��
��  �
G�C��f                                    Bxbnk~  �          @�33@�G��8Q��r�\��C��H@�G��L���w
=��(�C�Ǯ                                    Bxbnz$  T          A   @߮�^�R�b�\�ՅC�t{@߮�u�h�����
C�H                                    Bxbn��  
�          A
=@��\�Ϳ��8z�C���@��G
=�  �yG�C��f                                    Bxbn�p  
�          Aff@�(��0�׿B�\��C�W
@�(��%��p���\C��                                    Bxbn�  �          A
�HA{�7�����FffC�G�A{�0  �u��(�C���                                    Bxbn��  
�          A@��\��(����
���HC�)@��\�����L�����C�T{                                    Bxbn�b  T          A�@��H��ff?�  @�=qC�9�@��H����>L��?��
C��\                                    Bxbn�  "          A�
@����  ?��\@�=qC�S3@����(�>L��?���C��                                    Bxbn�  �          AQ�@�{��Q�?��\@�Q�C���@�{���H�\)�Tz�C��q                                    Bxbn�T  �          A33@�\��=q?�G�A(�C�1�@�\�Ǯ>��@p�C���                                    Bxbn��  �          A
=q@�\�h��@QG�A��\C�b�@�\���@$z�A��C��                                    Bxbo�  �          A�R@�p��dz�@^{A��\C��f@�p���(�@1�A��
C���                                    BxboF  T          A��@��H��Q�@>�RA��RC��@��H��
=@
�HA_�
C��H                                    Bxbo)�  �          AG�@�G���@#33A~�HC�N@�G�����?˅AG�C�\)                                    Bxbo8�  �          Ap�@��H���@�33A؏\C���@��H��
=@N{A�p�C���                                    BxboG8  �          A�H@�{�W
=@�{BffC�9�@�{����@�  A���C�1�                                    BxboU�  �          A�\@�=q���@eA���C�.@�=q���@�RAh��C���                                    Bxbod�  �          A�@�������@FffA���C��@������H?�G�A$z�C�5�                                    Bxbos*  �          A33@�  ��p�@;�A�C�!H@�  ���H?�
=AffC�B�                                    Bxbo��  �          A@�ff���@�\)Bp�C�S3@�ff��G�@�33A��C�޸                                    Bxbo�v  �          Az�@�Q����@�Q�A�(�C�#�@�Q���z�@\��A��\C�XR                                    Bxbo�  �          A33@�\)��  @�(�Bp�C��@�\)��{@�(�A�=qC��
                                    Bxbo��  �          Ap�@�Q���\)@��\A�z�C�g�@�Q���
=@`  A�
=C��
                                    Bxbo�h  T          A�R@����  @p��A�=qC��@����=q@�HAm�C��                                    Bxbo�  
�          A@�p���z�@eA�33C�o\@�p���@G�AaG�C�g�                                    Bxboٴ  
�          A�@�\)���@8Q�A�p�C���@�\)��?��A�C�0�                                    Bxbo�Z  �          A��@ȣ�����@XQ�A���C�|)@ȣ���G�@
�HAS
=C�Z�                                    Bxbo�   �          A{@����H@��A�{C��=@����@S33A���C��                                    Bxbp�  �          A@��
���R@���A��C��@��
��  @n�RA�(�C�Ǯ                                    BxbpL  �          A�\@�p���
=@�
=B�RC�T{@�p���=q@�33A��C��
                                    Bxbp"�  �          A��@��H���R@�  A�Q�C���@��H��(�@FffA�(�C�\                                    Bxbp1�  �          A  @�\)�Vff@�ffA�  C�/\@�\)��p�@\)A��C���                                    Bxbp@>  �          A  @ᙚ���@�AyG�C��@ᙚ��33?�Q�A=qC��                                    BxbpN�  �          A\)@����
=@.�RA�p�C�:�@����z�?��HA'�C�+�                                    Bxbp]�  �          A��@�G����
@C33A���C���@�G����@Q�AM��C�`                                     Bxbpl0  �          A�\@У���=q@dz�A��C��@У���z�@=qAl(�C��f                                    Bxbpz�  T          A=q@޸R��p�@S33A��HC���@޸R��ff@G�Ab{C�T{                                    Bxbp�|  T          A��@�G��}p�@��RB	�HC�c�@�G����\@��
A�Q�C���                                    Bxbp�"  �          A=q@mp���ff@�BP33C��)@mp����@�  B-p�C��                                    Bxbp��  �          A��@33�|(�A��Br�C�!H@33���@�ffBK�C��H                                    Bxbp�n  �          A��?�\)�l��A�HB��3C�h�?�\)���@��BZ��C�                                    Bxbp�  �          AQ�?�Q��h��A��B~(�C��?�Q����@�{BUC�P�                                    BxbpҺ  �          A�?�33��p�A�RBq  C���?�33���R@��BGp�C���                                    Bxbp�`  �          Ap�@Tz���{@�BFQ�C�Q�@Tz��Ϯ@�33B�RC���                                    Bxbp�  �          Az�@�����@��B8p�C���@���Ӆ@�33B�RC��                                    Bxbp��  �          AG�@fff��Q�@ÅB$ffC�y�@fff��\@��\A��\C�}q                                    BxbqR  �          A@�  ��\)@�(�AڸRC�aH@�  ���@5A�\)C���                                    Bxbq�  �          A�@X�����@ӅB9�C�C�@X����  @�ffB�\C��                                    Bxbq*�  �          A�H@s�
��Q�@�
=B(�C�+�@s�
���@�{A�\C�>�                                    Bxbq9D  �          Aff@��\��Q�@��\A��C��{@��\���H@`��A�=qC�%                                    BxbqG�  �          A�\@�����@ʏ\B*
=C��{@����  @��B�\C���                                    BxbqV�  T          A�@�{��33@���B ffC���@�{���@[�A��C�(�                                    Bxbqe6  �          A(�@hQ���ff@�A�C��f@hQ���{@>{A�  C��                                     Bxbqs�  �          A�
@z=q��=q@��HA�(�C�~�@z=q��G�@7
=A�\)C�XR                                    Bxbq��  �          Az�@��
���
@��\A�\C�@��
����@N�RA���C�p�                                    Bxbq�(  �          A�@�����
@�=qA��
C���@���ָR@c33A���C��                                    Bxbq��  �          Aff@����G�@���A��
C���@����
=@-p�A�33C�Ff                                    Bxbq�t  �          A{@��R��G�@[�A�
=C��@��R���H@   AQC���                                    Bxbq�  �          A�R@��H��
=@u�Ȁ\C�q�@��H��33@��A{�C�K�                                    Bxbq��  �          A�@c�
��\)@�{B�
C�^�@c�
����@xQ�A�ffC��H                                    Bxbq�f  �          A
�R@\)��{@�Q�B'=qC�Y�@\)��\)@�{A���C���                                    Bxbq�  �          A\)@�����p�?z�H@أ�C���@�����
=��G��AG�C�~�                                    Bxbq��  �          A��@�\)���R?�z�A  C�N@�\)���
>aG�?�
=C��                                    BxbrX  �          A��@������?�@�\C�o\@�����33���
���HC�+�                                    Bxbr�  �          AG�@�����녽��
���C�!H@��������\)�
=C�y�                                    Bxbr#�  �          A
=@Ǯ�˅�
=q�\��C�:�@Ǯ���
��=q�<��C��=                                    Bxbr2J  �          A@�Q���33�!G�����C�AH@�Q���=q�z��X  C��                                    Bxbr@�  �          A��@����ۅ��
=��{C��@�����
=�#�
��\)C��                                    BxbrO�  �          A�@Å��33��
=�=qC��@Å���*=q��G�C���                                    Bxbr^<  �          A��@ڏ\��p���z���C��@ڏ\��=q�  �mG�C���                                    Bxbrl�  �          A��@�����
������p�C�f@�������Q��`��C���                                    Bxbr{�  �          A
�R@�  ����G���{C���@�  ���Ϳ�p��8��C�b�                                    Bxbr�.  �          A
�R@�33���H>aG�?���C��)@�33�����#�
���C��q                                    Bxbr��  �          A�
@���k��O\)��z�C�H�@���[��Ǯ�&�HC��                                    Bxbr�z  |          A��@����Mp��.{��G�C�q�@����$z��U���  C���                                    Bxbr�   �          A=q@�33��녾�33���C��
@�33���
��G����C�w
                                    Bxbr��  �          A��@�Q���=q@A�A�{C�W
@�Q��ҏ\?�
=A*�HC�4{                                    Bxbr�l  �          A�
@����=q@Q�A�{C���@�����
?���A;�C��{                                    Bxbr�  �          A��@����p�@(��A���C�AH@����=q?�\)@���C�t{                                    Bxbr�  �          A�@�{��@%�Ax��C��3@�{��?�G�@���C�5�                                    Bxbr�^  �          A�@�\)�׮@|��A�Q�C�,�@�\)��@�HAg�C���                                    Bxbs  �          A��@�����p�@���A��C���@�����z�@%A|  C�z�                                    Bxbs�  �          A{@\��ff@l(�A��\C���@\���H@{AU�C�P�                                    Bxbs+P  �          A��@����R@��\B�C�h�@����@�ffA�ffC���                                    Bxbs9�  �          A  @���n�R@���BR  C��@����Q�@љ�B.��C���                                    BxbsH�  �          A��@~�R���@��B�C�q�@~�R��@|��A�\)C���                                    BxbsWB  �          A(�@z�H�ə�@�\)B�C��3@z�H��\@\)A�33C�\                                    Bxbse�  �          A33@���љ�@�
=B33C��f@����\)@[�A���C�C�                                    Bxbst�  �          A�
@���Å@��
A���C�@ @�����@Z�HA��RC�XR                                    Bxbs�4  �          A��@�33�Ӆ@�33A߅C�&f@�33��z�@333A���C���                                    Bxbs��  �          A�@���У�@��\A�=qC��f@����Q�@#33A��RC�q�                                    Bxbs��  �          A��@l(���
=@}p�A�=qC���@l(����@�Aj=qC���                                    Bxbs�&  �          A��@��\��
=@��A��
C�*=@��\��{@�Ay�C��
                                    Bxbs��  �          A{@��
�ƸR@n�RA�(�C��@��
��(�@G�Ae��C���                                    Bxbs�r  �          A��@����=q@J=qA�\)C�Ff@����33?�=qA (�C�B�                                    Bxbs�  �          AG�@�ff���@g
=A�=qC���@�ff��G�@33AJ�HC�c�                                    Bxbs�  T          A33@�ff��G�@�33A�z�C�H�@�ff�ٙ�@)��A�
=C��R                                    Bxbs�d  �          A
=@�����33@q�A�\)C�N@�������@
=AN�\C�                                      Bxbt
  �          A�R@����  @s33A�G�C�e@����@��A^�\C��)                                    Bxbt�  �          A@�\)�Ǯ@~�RA�\)C�E@�\)��
=@�RAv{C���                                    Bxbt$V  �          A��@������@`��A��HC��f@�����{@   AF=qC�,�                                    Bxbt2�  T          A=q@�
=��{@5A��RC��q@�
=���?�G�@��RC��                                    BxbtA�  T          A=q@�(����
@#33A|z�C�c�@�(�����?}p�@���C���                                    BxbtPH  �          A�R@�Q��љ�?�A$(�C�e@�Q���Q�>\)?aG�C��3                                    Bxbt^�  �          A=q@��R�ʏ\@��B=qC�T{@��R���@y��A��
C�9�                                    Bxbtm�  �          A
=@�=q��=q@�G�B��C��3@�=q��@n{A�C��\                                    Bxbt|:  �          Ap�@�(����@�33A��C�{@�(���(�@8Q�A�{C��f                                    Bxbt��  �          AG�@�z����@�33AĸRC�y�@�z����@ffA\��C�                                      Bxbt��  �          A�@�����z�@��A�(�C���@�����(�@\��A�\)C��\                                    Bxbt�,  �          A=q@���θR@��B�C���@�����H@���A�{C�w
                                    Bxbt��  �          A33@�
=��Q�@�  B!�\C���@�
=��G�@�{A���C��                                    Bxbt�x  �          AQ�@����ȣ�@��BQ�C�f@�����=q@h��A�{C��                                    Bxbt�  T          A�@������
@�Q�A��C���@������@I��A�\)C��f                                    Bxbt��  �          A z�@�G��ə�@�(�A�  C�w
@�G���ff@B�\A��C���                                    Bxbt�j  
�          A�H@��H���@<(�A�\)C�+�@��H��(�?��RA�C��3                                    Bxbu   �          A�@�=q��ff@޸RB133C��@�=q���R@�\)BG�C��=                                    Bxbu�  �          A�@�{����@�33B�HC���@�{���@�G�A�(�C���                                    Bxbu\  �          AG�@�����p�@��B��C�!H@������@�z�AC���                                    Bxbu,  �          A
=@θR����@�(�B	��C�H@θR��\)@���A��
C�f                                    Bxbu:�  �          A�H@�\)���@��B\)C�  @�\)��\)@��RAȸRC�                                    BxbuIN  �          Aff@�{����@�A��C�N@�{�ə�@b�\A���C���                                    BxbuW�  �          A��@�  ���?��
A�\C�G�@�  ��(����E�C��\                                    Bxbuf�  �          A�@�
=��(���\�W
=C��=@�
=��=q�G��Up�C�=q                                    Bxbuu@  �          A��@�{��p�@
�HAc
=C���@�{��Q�?�@mp�C��                                    Bxbu��  �          A�\@����?���A?
=C�L�@����>�  ?У�C���                                    Bxbu��  �          A�\@����ʏ\?�33A*�\C�Ф@�����G�=��
?   C�Y�                                    Bxbu�2  �          A�@�����\?�\)@�C�ff@�������
��
C�*=                                    Bxbu��  �          Az�@����녽#�
��  C�� @����(���  ��HC�3                                    Bxbu�~  �          A�@�G���ff����Y�C���@�G���  �e���p�C�h�                                    Bxbu�$  �          Ap�@�  �ȣ��G��M�C�s3@�  ���H�^�R��z�C��                                    Bxbu��  �          A��@�\)�׮��\)�)p�C��\@�\)��z��Mp����RC��                                     Bxbu�p  �          A��@��H����Q��C��f@��H�����z��z�C�B�                                    Bxbu�  �          A��@�\)��{=�?G�C���@�\)���H�h����C��                                    Bxbv�  �          A
{@�(��\(�?�@c33C�5�@�(��^{�����
�HC�!H                                    Bxbvb  �          A�RA Q��u?\)@dz�C�33A Q��w
=����'
=C�%                                    Bxbv%  �          Az�@��R���R>�p�@�C�@��R���.{���C�(�                                    Bxbv3�  �          AQ�@�
=�j�H�\)�h��C���@�
=�aG������=qC��                                    BxbvBT  |          A�\A�H�7�������C��RA�H�-p���  ��ffC�R                                    BxbvP�  T          A{A33�(Q�:�H���HC��
A33����{�	G�C�e                                    Bxbv_�  �          A33Az��z῱��\)C���Az��33���Ap�C��                                    BxbvnF  �          A�A{�/\)���0��C�k�A{����(��n�HC��                                    Bxbv|�  �          AffA	��I����G��-�C��3A	��'
=�   �w�
C���                                    Bxbv��  �          A��A	���g
=���R�@z�C��)A	���@  �5���{C�e                                    Bxbv�8  �          A�HA ���N{���X  C�
A ���%�4z���C��                                    Bxbv��  �          A\)@��R�   �
=�|��C�G�@��R�����:�H��33C���                                    Bxbv��  �          A�RAz��ff�/\)��\)C��)Azῆff�G
=��\)C�b�                                    Bxbv�*  �          A�RAQ��Q��:�H���RC�ffAQ�c�
�P  ��Q�C��                                    Bxbv��  �          A�A�H�   �4z���ffC�U�A�H���H�O\)��z�C��                                    Bxbv�v  �          A  Aff��G��|���ˮC��HAff��ff��ff���
C�h�                                    Bxbv�  �          A�HAp���=q���
�ՅC�Q�Ap��u���\��
=C�*=                                    Bxbw �  �          A�R@޸R�0�����\)C�0�@޸R?:�H���33@��                                    Bxbwh  T          A��@�\)����{���C���@�\)<#�
���H����=L��                                    Bxbw  �          A�A����\��{�Q�C���A�Ϳ�\)��z��?�C��                                    Bxbw,�  �          A(�A��#�
�@  ��p�C�Y�A��녿���
=C�'�                                    Bxbw;Z  �          AAQ��\)��(��\)C��AQ���\������C�<)                                    BxbwJ   �          A(�Aff�#�
���@  C�c�Aff�(��J=q��z�C���                                    BxbwX�  �          A  A��\)�����C�Q�A��
=��  �*�\C�t{                                    BxbwgL  �          A�
A{�S33����p�C�˅A{�E���G���C�l�                                    Bxbwu�  �          Az�A=q�Q녿\)�a�C�W
A=q�AG�����{C��                                    Bxbw��  �          A��@�(���(��u���{C�  @�(��c�
��
=���
C��f                                    Bxbw�>  �          A��A���R�:=q��\)C���A������X����G�C�H�                                    Bxbw��  �          A
�\@����p��QG����RC��@�����
�^�R����C���                                    Bxbw��  �          A��@����S33?u@��HC���@����[�=L��>\C�aH                                    Bxbw�0  �          A
�H@ᙚ�L��@x��A��C��=@ᙚ���
@9��A�Q�C��q                                    Bxbw��  
�          Aff@�33�XQ�@��
A�G�C�G�@�33��33@E�A�(�C�AH                                    Bxbw�|  T          A��@���\��@��B �
C���@����33@mp�A£�C��
                                    Bxbw�"  �          A(�A z����R?�{@��
C�xRA z���33���E�C�)                                    Bxbw��  �          A{@�����?�(�AQ�C��
@���Q�>�=q?�(�C�/\                                    Bxbxn  �          A{@�G����?�G�A2{C�e@�G���p�?�@S33C��=                                    Bxbx  �          A�@�(���
=@(��A�33C�<)@�(���Q�?��\A�RC��=                                    Bxbx%�  �          Ap�@���\)@!G�A��RC���@����?�(�@�ffC�&f                                    Bxbx4`  �          A  @��H���R@AG�A���C�3@��H���
?�G�A5�C�7
                                    BxbxC  �          A�
@�  �u�@   A��RC��
@�  ���
?�{A\)C�N                                    BxbxQ�  �          AQ�@�����H@%�A���C�b�@����z�?�\)A(�C��                                    Bxbx`R  �          A33@�=q�s33@.{A��C�L�@�=q��z�?�=qA(  C��                                    Bxbxn�  �          A
ff@��l��@33Aw\)C�@���{?���A   C�L�                                    Bxbx}�  �          AQ�@�33�e?�ffA"�HC��{@�33�xQ�?�\@Tz�C�ٚ                                    Bxbx�D  �          A�@�33�n�R?��HA2ffC�O\@�33����?!G�@���C�Y�                                    Bxbx��  �          A��@��H�p��?��\Az�C�/\@��H�}p�>8Q�?���C��
                                    Bxbx��  �          A�@�33�c�
?aG�@�z�C�\@�33�j�H�\)�uC��)                                    Bxbx�6  �          A�H@��
�`  ?.{@��C�J=@��
�c33���
�{C�!H                                    Bxbx��  �          A�
@��i��>��R@Q�C�Ф@��e�8Q���  C�                                      BxbxՂ  �          A�@��^�R>�  ?�Q�C��@��Z�H�:�H��ffC��                                    Bxbx�(  .          @�33@�(������)����
=C�]q@�(��G��j=q��Q�C�}q                                    Bxbx��  �          @��@�����ff�5��
=C��=@����n{��G����C�                                      Bxbyt  .          @��@�����\�Vff�̏\C�.@���l������33C��                                    Bxby  T          @��@ ����(������E��C��=@ ���Q��ҏ\�x�HC�C�                                    Bxby�  �          @���@�  �S�
�g
=��\C�f@�  �
�H�����ffC��\                                    Bxby-f  .          @��@�(���G��vff��33C�
=@�(��B�\������C���                                    Bxby<  �          @�p�@�Q��q��_\)��C��@�Q��(����p���
C�&f                                    BxbyJ�  �          @�\)@�G��W
=�`  ���HC���@�G���R����=qC�9�                                    BxbyYX  �          A (�@��H�e��j�H���
C���@��H�����G��
��C�N                                    Bxbyg�  �          A ��@�p��b�\������(�C���@�p��p���\)���C��H                                    Bxbyv�  �          A�@�p���ff����{C�s3@�p��/\)��  �-  C��                                    Bxby�J  T          A Q�@��\���\��\)��C��@��\�QG�����=�
C�s3                                    Bxby��  T          A ��@��
���H��{�
=C�&f@��
�3�
����6p�C���                                    Bxby��  �          A  @��
�hQ������C�#�@��
�p���G�� �C���                                    Bxby�<  �          A�@����h�����	p�C��3@�����������'=qC��                                    Bxby��  �          A@��H��������C�3@��H�$z���33�2z�C�#�                                    BxbyΈ            A=q@����z���=q�=qC���@���8Q������)\)C�#�                                    Bxby�.            AG�@��
��������p�C�` @��
�u�����&C��f                                    Bxby��  T          A�@��\��\)�����=qC�'�@��\�������
�*z�C��
                                    Bxby�z  �          Ap�@�\)������\�뙚C��f@�\)��p�����*�C��)                                    Bxbz	             A�@�������~�R��C�=q@����\)�����%�
C�q                                    Bxbz�  �          A��@�����G��r�\�؏\C��q@�����p���z��!\)C�XR                                    Bxbz&l  �          A�@:�H����
=�=qC�<)@:�H���H�����=�RC��{                                    Bxbz5  �          A\)@��R��ff������(�C��@��R�ʏ\�C�
����C�aH                                    BxbzC�  �          Ap�@����ڏ\�Q����C�y�@�����G��333��  C��f                                    BxbzR^  T          AQ�@�Q���(��k����C�
@�Q���=q�5����HC�^�                                    Bxbza  T          A��@�33��z�.{��=qC�g�@�33�����"�\����C��=                                    Bxbzo�  T          A�\@�\)��=q�����%�C�&f@�\)������s
=C�{                                    Bxbz~P  �          A�\@������H?�  @�33C�K�@������H��  ��=qC�K�                                    Bxbz��  �          A�@������@:=qA�
=C���@������?z�H@���C�4{                                    Bxbz��  �          A�\@Å�˅?���@��C���@Å��p��@  ���C��=                                    Bxbz�B  �          A�@���G�@*=qA��\C��3@��ʏ\?O\)@��\C���                                    Bxbz��  �          A	�@�����@%�A�Q�C�P�@����=q?8Q�@�=qC��                                    Bxbzǎ  T          A	��@�=q���ÿ�G����C�>�@�=q����2�\��  C��                                    Bxbz�4  �          A33@��ÿ��
�����V��C�G�@���?c�
�����Z��A�R                                    Bxbz��  �          A33@���=q����;ffC�q@�?!G���R�@�@��H                                    Bxbz�  �          Aff@u����H�m�C�N@u=�\)��ffu�?�                                      Bxb{&  �          A�@�G��@  �����YG�C��@�G��&ff��33�t\)C���                                    Bxb{�  �          A�H@���B�\���UC�ff@�녿:�H��ff�q��C�*=                                    Bxb{r  �          A(�@����;����H�[=qC��H@��׿\)��z��u�C�J=                                    Bxb{.  T          A
=@��
�Y������G��C��
@��
��33�����f�C��q                                    Bxb{<�  �          A
=q@�{�C�
�����<�HC�R@�{�}p���p��W�RC�N                                    Bxb{Kd  �          A  @��H�@����33�M=qC���@��H�W
=��ff�i�RC��=                                    Bxb{Z
  T          Aff@��R�^�R�ƸR�>�RC�@��R��33��\)�`C�G�                                    Bxb{h�  �          A�R@����7����
�9�C���@��ÿk��θR�T�C���                                    Bxb{wV  �          A@�z��G
=��(��0�C�@�z῜(���=q�M��C��)                                    Bxb{��  T          A��@��2�\��z��4=qC��
@��Tz��θR�K�HC�s3                                    Bxb{��  �          A��@���;���=q�:�C���@���fff��p��T�RC��=                                    Bxb{�H  �          A�H@����%�У��K
=C�w
@��;�(���  �a{C�XR                                    Bxb{��  �          A	p�@����\)���W��C�9�@��;W
=���l�C���                                    Bxb{��  �          A(�@�ff�
=��\)�F��C�>�@�ff�aG���z��Y�C���                                    Bxb{�:  �          A(�@s33�\)����iffC��R@s33=�G���Q��|�?У�                                    Bxb{��  �          A{@�(������G��[C�^�@�(�>�{��G��h(�@��R                                    Bxb{�  �          A��@�녿���ff�M{C�
@��>L����\)�Y�
@��                                    Bxb{�,  �          Ap�@�(��z������`��C���@�(�>W
=��R�p�\@6ff                                    Bxb|	�  �          A��@:=q�
=���
�|=qC��H@:=q=�Q�����q?�                                      Bxb|x  �          A��@:=q��=q�����C���@:=q?#�
��
=��AG
=                                    Bxb|'  �          AQ�@Mp��p����r�C�AH@Mp��u���H33C�z�                                    Bxb|5�  �          A��@U��	����Q��uz�C��=@U�>����=qff@�Q�                                    Bxb|Dj  �          A	��@�\�����
��C�` @�\?������HB��                                    Bxb|S  �          A
ff?��H������C�h�?��H?У��Q�B�                                    Bxb|a�  �          A
�\?�33��(���\� C��H?�33?��
��)B*�H                                    Bxb|p\  �          A�R?���=q� Q��\C�8R?�?�������3A�33                                    Bxb|  �          A�?�녿�ff��HaHC���?��?�33�  G�Bz�                                    Bxb|��  �          A
�\>k��xQ��	p�¦�
C���>k�?����33#�B��                                    Bxb|�N  �          A�
?(�ÿ(���R§��C��{?(��@����R\)B���                                    Bxb|��  �          Az�?z�H�.{��H¦{C�.?z�H@�R����.B��q                                    Bxb|��  �          A��aG�>W
=� ��¯��C	aH�aG�@333��G�.B�.                                    Bxb|�@  �          A�R�W
=>�  ���§��C#aH�W
=@7
=��\BԽq                                    Bxb|��  �          AG��#�
>�� ��²�B�z�#�
@/\)��=q8RB���                                    Bxb|�  �          @��
?\(�=�\)���¦��@��\?\(�@!���(��B�\)                                    Bxb|�2  �          @�(���?:�H����¨aHB�aH��@Dz�����p�B��f                                    Bxb}�  �          @�(�>�녾Ǯ���
©ffC�U�>��?У����33B��{                                    Bxb}~  �          @�
=�6ff�����33�C>�
�6ff?�p������}�\C�                                     Bxb} $  �          @�\)��(�>.{����ǮC,:῜(�@Q���(���B�q                                    Bxb}.�  �          @�z��#33?W
=��
=�)C!��#33@:�H���H�az�C&f                                    Bxb}=p  �          @�ff@�33�aG��o\)�z�C��@�33>����tz��(�@�
=                                    Bxb}L  �          @��@��H>�����D��@�p�@��H@�������/�Ạ�                                    Bxb}Z�  �          @�
=@�  �
=��z��M�HC��@�  ?�p������H(�A��R                                    Bxb}ib  �          @�
=@��
>�������833?�@��
?�p���\)�(�
A�=q                                    Bxb}x  �          @���@���>8Q���(���@ ��@���?���u���A��                                    Bxb}��  �          @�Q�@ۅ�E�>�ff@mp�C���@ۅ�aG�>.{?�\)C�Y�                                    Bxb}�T  �          @�{@�Q�:�H?�=qA0z�C���@�Q쿐��?�G�A��C�B�                                    Bxb}��  �          @�@��
���>���@\��C�>�@��
��>u@�C��f                                    Bxb}��  �          @�=q@�  ��
=>�ff@�z�C�@�  ���>�z�@&ffC�y�                                    Bxb}�F  �          @�p�@��R�u?   @���C���@��R�Ǯ>Ǯ@n�RC�q                                    Bxb}��  �          @�
=�
=?����
�w�HC(��
=@g����={B�L�                                    Bxb}ޒ  T          @���8Q�?�\)��p��sQ�C�\�8Q�@aG������?p�C=q                                    Bxb}�8  �          @�\)�	��@���ff�HC���	��@�=q��=q�D�B��                                    Bxb}��  �          @��R�,(�@p���  �z��C�f�,(�@������>\)B�aH                                    Bxb~
�  �          @��
�|(�?�33���ZC5��|(�@xQ���p��,(�Ch�                                    Bxb~*  �          @���*�H?�Q��ȣ�\)C�=�*�H@_\)�����K�B���                                    Bxb~'�  �          @�=q?���aG��
=�}p�C��?��?�\��
�r��A��                                    Bxb~6v  �          @�Q�@�=q�#�
@�p�BB\)C���@�=q���@�p�B(=qC��3                                    Bxb~E  �          @�
=@�zᾸQ�@��B1�C��@�z��p�@���BffC���                                    Bxb~S�  �          @���@�R?8Q�?�AI�A�z�@�R>��H?=p�A�33AG�
                                    Bxb~bh  �          @���(Q�>���|���aQ�C0���(Q�?�  �j=q�I�CG�                                    Bxb~q  �          @�  ���#�
��ff�{C4:��@p�����Ch�                                    Bxb~�  �          @��;�����\)�{C6G��;�@���z��k33C@                                     Bxb~�Z  �          @陚�2�\?8Q���{�C%���2�\@3�
����ZG�C��                                    Bxb~�   �          @�{@w
=���@�p�BO\)C�޸@w
=��@��B;ffC�B�                                    Bxb~��  �          A@�  �Q�@�  B8  C��q@�  �0��@��B��C���                                    Bxb~�L  �          Az�@���>�p�@ϮBP��@�p�@��׿�z�@�
=BD�C��)                                    Bxb~��  �          @�\)@��?u@�\)BS=qAK�@�����@��RBR  C���                                    Bxb~ט  �          A{@�z�?�@�(�B_=q@�33@�z��z�@�(�BSp�C���                                    Bxb~�>  �          A(�@��>��@�(�BW
=?�ff@���
�H@ȣ�BF�RC���                                    Bxb~��  �          A33@�  ��(�@ҏ\BM(�C���@�  �,��@�Q�B5��C�b�                                    Bxb�  �          @�=q@��H?�\)@��B@�\A|Q�@��H�.{@�\)BF=qC��)                                    Bxb0  T          @�  ?p��@%?uA��B�.?p��@?�B{B���                                    Bxb �  �          @�33�4z�@  �Dz��!�C\)�4z�@L(������CxR                                    Bxb/|  T          @��\�l��>��
��(��@33C/  �l��?�  �p  �*
=C�)                                    Bxb>"  �          @��
��p�?�  ��=q�=(�C"33��p�@Q���ff��C��                                    BxbL�  �          @���p�@?\)����/(�CY���p�@�p��z�H��\)C��                                    Bxb[n  �          @������\@33���\�+\)C\���\@mp������z�C�)                                    Bxbj  �          @�����=q?xQ���33�2Q�C(����=q@(Q���{�Q�Cn                                    Bxbx�  �          @��
��G�?�\)����({C"=q��G�@.{�S�
�=qC��                                    Bxb�`  �          @�ff��33?�����
�p�C��33@HQ��Mp���{C��                                    Bxb�  �          @�{��G�?�\)�\���\)C T{��G�@0  �)����{C�
                                    Bxb��  �          @�ff��\)@4z��E�ۅC����\)@o\)�����(�CxR                                    Bxb�R  �          @ʏ\��\)@|(����z�C	
=��\)@��׿G���{C!H                                    Bxb��  �          @����p�@�Q�L�Ϳ�B���p�@�ff?޸RA��B�\)                                    BxbО  �          @ʏ\���@�p����
�b�HB�k����@�33?��@�G�B�L�                                    Bxb�D  �          @��H��33@��\��(���z�C+���33@��\)��=qC�
                                    Bxb��  T          @�  �_\)@w
=�����{C(��_\)@�{�\)��G�C��                                    Bxb��  �          @�=q�I��@^�R�����ffC(��I��@z�H�   ��z�C �                                    Bxb�6  �          @��H���@a�����(�B�(����@��
�Y���$  B�z�                                    Bxb��  �          @��ÿ�@QG�� ����p�B��f��@q녿0���$  B�G�                                    Bxb�(�  �          @n�R>��@Tz��=q���B��f>��@j=q��=q��Q�B�aH                                    Bxb�7(  �          @X��@
=q?�\)�����\)B#Q�@
=q@�R���'\)B7p�                                    Bxb�E�  �          @u@\)@,�Ϳu�j{B=\)@\)@7�=�Q�?��BD{                                    Bxb�Tt  �          @5�?��@�R�z��>{B��\?��@!G�>�p�@��HB�=q                                    Bxb�c  �          @�z��AG�@�(���\)�>�RB�B��AG�@��?��@�=qB���                                    Bxb�q�  �          @�(���p�@�zῼ(��2�HC���p�@���>��?�{C��                                    Bxb��f  �          @�\)�У�@\)�����=qC��У�@�(�>��@C�
C��                                    Bxb��  �          @��\��ff@�(��
=q�}p�C�{��ff@���?���A�Cz�                                    Bxb���  �          @������H@�(��\)���
C!H���H@���?���A\  C	��                                    Bxb��X  �          @�
=��{@�\)��ff�W
=C����{@�33?���A33C�=                                    Bxb���  �          @������@�
=>�
=@Z=qCn����@�
=@�RA�(�C
}q                                    Bxb�ɤ  �          @�����33@�\)?#�
@�Q�CQ���33@��@'�A��C	                                    Bxb��J  �          @��x��@�{>��R@ffB�G��x��@��
@*�HA�  B�{                                    Bxb���  �          @�=q>���@�ff��  �/33B���>���@�ff?�p�A,(�B���                                    Bxb���  �          @�{@�z�@(���U�G�A���@�z�@k�����ffB�                                    Bxb�<  �          @��R@aG�@�R�|���)�Bz�@aG�@qG��/\)��  B;��                                    Bxb��  �          @�33?��H@ff�s�
�R�BW�H?��H@e�*=q�33B�=q                                    Bxb�!�  �          @���?��\@5���{�j��B�B�?��\@��R�a��ffB�33                                    Bxb�0.  �          @�{?�@:�H��33�U{B�(�?�@��\�<���z�B��q                                    Bxb�>�  �          @�ff?�=q@�
=��G��$
=B���?�=q@ə��ff����B���                                    Bxb�Mz  �          @��H@�@�{�z=q��\B�.@�@�Q�˅�Q��B��{                                    Bxb�\   �          @�=q@���@����3�
�ĸRBD�@���@��
�\(����HBT�                                    Bxb�j�  �          @�Q�@2�\@���ff�!�Bg�@2�\@����p���  B�k�                                    Bxb�yl  �          @�
=��R@�p��G���Q�B����R@�{?#�
@�  B׏\                                    Bxb��  �          @�(����
@���?�{AAB��)���
@�@e�B  B���                                    Bxb���  �          @�
=��\@�����
�C�
B۔{��\@��?�A�(�Bݙ�                                    Bxb��^  �          @��ͿB�\@أ׿=p��ƸRB��)�B�\@љ�?�\)A}�B�G�                                    Bxb��  �          @��>k�@���>���@/\)B��>k�@���@7�A��
B��R                                    Bxb�ª  �          @�
=@�33����@�p�BJG�C�e@�33�n�R@��
B�
C�z�                                    Bxb��P  �          A�@�33�w
=@ϮB'�C�ٚ@�33����@�G�A�33C���                                    Bxb���  �          A&=q@�
=�Fff@�B8
=C�7
@�
=��{@�ffB�
C�U�                                    Bxb��  T          AQ�@��\��=q@��B[�C��@��\��ff@ָRB2��C��H                                    Bxb��B  �          A�@�{�ٙ�@�z�BV  C��@�{��z�@�B)  C��                                    Bxb��  �          @��\@���	��@���B<ffC�,�@����{@�Q�BC��H                                    Bxb��  �          @�G�@k����@�(�Bb��C��@k��"�\@���BA��C��R                                    Bxb�)4  �          @ҏ\@O\)?�=q@��Bj�
A�33@O\)��z�@��HBi��C�+�                                    Bxb�7�  �          @���?�Q�?�@�p�By�\BM�
?�Q��@���B�k�C�w
                                    Bxb�F�  �          A*�\@�{�N{@�Q�B1�
C���@�{���@�{B Q�C��                                    Bxb�U&  �          A0z�@�Q��Fff@�p�B7�
C�\@�Q����@��
B�HC��                                    Bxb�c�  �          A.=q@����AG�@�p�B;\)C�  @������H@�z�B	��C���                                    Bxb�rr  �          A-��@�{�,��@��B?��C���@�{���\@ʏ\B�C��                                    Bxb��  �          A((�@��
�&ff@�  BA�C��H@��
����@�z�BG�C��)                                    Bxb���  T          @�(�@�(��Q�@��B+\)C�/\@�(���G�@��
A��
C���                                    Bxb��d  
�          @�Q�@ ��?:�H@uBsQ�A��@ �׿Q�@tz�Bqp�C��R                                    Bxb��
  
�          @��R�AG�@�33?!G�@ᙚB��q�AG�@a�@
�HA�z�C��                                    Bxb���  �          @�ff���@G
=��p����Cp����@���(�����C�3                                    Bxb��V  �          A����
=@-p���p��  C�{��
=@�
=�AG���p�C�                                     Bxb���  �          A
ff����@�  ��ff��Cn����@����'
=���
B��q                                    Bxb��  �          @����ff?��H�Mp����C!Ǯ��ff@C�
�(���
=C�\                                    Bxb��H  �          @�  ��@^{����.�CW
��@��R�X����p�C                                     Bxb��  �          @����q�@e�����?��C���q�@���y����
=B�p�                                    Bxb��  �          @��H����@K�����"\)CT{����@�33�QG���p�C��                                    Bxb�":  �          A���=q@$z���p��/=qC:���=q@��H��ff���C8R                                    Bxb�0�  �          Aff���\@C�
��(��;�
C�����\@�
=����33C�=                                    Bxb�?�  �          A�����
@9���Ǯ�CffC����
@��
��33��C�                                    Bxb�N,  �          A����H@>{��Q��Cz�C����H@�ff���\��
Cu�                                    Bxb�\�  T          @��
��z�=�Q���
=�ez�C2����z�@(������I
=CxR                                    Bxb�kx  �          @�ff�Vff?�(�����t�RCG��Vff@\)�����:�HC{                                    Bxb�z  �          @���&ff=#�
��Q��z33C3#��&ff@G����H�Y=qC.                                    Bxb���  �          @�ff�"�\>k������w��C.�\�"�\@
=�����Q�
CT{                                    Bxb��j  �          @����"�\?
=q�����=C(
=�"�\@,(���33�R�Cp�                                    Bxb��  T          @��=#�
�@  �
=q��C�j==#�
��33�K��l��C���                                    Bxb���  �          @��H�C33��p��θR�|(�CI�q�C33?�Q���p��y�C�f                                    Bxb��\  �          A z��}p�?��\��33�k  C"(��}p�@y�������8p�Ck�                                    Bxb��  �          Aff��ff@ff���Y�
Cn��ff@��H�����
C��                                    Bxb��  �          Ap����@�Q���(��2�\C�����@�\)������HC)                                    Bxb��N  �          A���G�@|(������%ffCO\��G�@��o\)�ɮC��                                    Bxb���  �          A����z�@g���
=�C�C}q��z�@Å��� �C ��                                    Bxb��  �          A�����
@,(���
=�PffC^����
@������  C�q                                    Bxb�@  �          A
=���@b�\�׮�;\)C�����@��R��  ��p�CT{                                    Bxb�)�  �          Ap���p�@W
=��Q��2��CaH��p�@�{�����RC��                                    Bxb�8�  �          A�
���\@?\)���
�9Q�C�)���\@�  ���\�ffC��                                    Bxb�G2  �          Ap�����?��R��p��S=qCT{����@������ 
=C	�
                                    Bxb�U�  �          A�����?���=q�P  C!�����@����(��   C�R                                    Bxb�d~  �          A�
��
=@p���R�H�C��
=@�Q����\�ffC	c�                                    Bxb�s$  �          A(���G�@.{����C(�Cٚ��G�@�ff���\�G�C�                                    Bxb���  �          A{��=q@(����I\)C@ ��=q@�=q��
=��\C	�
                                    Bxb��p  �          A���Q�@ ����
=�C�HCs3��Q�@�\)���\���C	�=                                    Bxb��  �          A  ���
@\)��p��A�\C&f���
@��\�����C	�H                                    Bxb���  �          A	����?�(���R�offC#�����@�����=q�<ffC
                                    Bxb��b  �          Az���p�?�Q���\�]�C"����p�@�ff��(��,p�C
\                                    Bxb��  T          A{���?�
=��=q�^Q�C#aH���@�����33�-��C
5�                                    Bxb�ٮ  �          A�\���R���
�  z�C7=q���R@R�\����B�                                    Bxb��T  �          A�����>�\)��8RC-=q���@`  ���
�l�B���                                    Bxb���  �          Aff�'�>k���\)ffC.���'�@W���ff�i��B��                                    Bxb��  �          @���+������
={C<Ǯ�+�@,(��߮�u��C�R                                    Bxb�F  �          AG��:=q�
=��Q��)C?z��:=q@"�\���H�u��C
ٚ                                    Bxb�"�  
�          @�33��R��G����=CLG���R@����{� C@                                     Bxb�1�  �          A���R������  p�C^����R?����HG�Ck�                                    Bxb�@8  �          A=q��\�����RB�C[0���\?�z����RW
C�                                    Bxb�N�  �          A\)�:=q�W
=���L�C8
�:=q@C33���oG�C��                                    Bxb�]�  �          A  �Mp�>8Q�����{C0���Mp�@XQ���Q��a�C��                                    Bxb�l*  �          @��?�����K�ffC�?�>��
�UG�A��                                    Bxb�z�  �          @u�����=L�Ϳ���&p�C2�����?z῔z���CQ�                                    Bxb��v  �          @�p�>�p��E��:=q�,�C�w
>�p��˅�{��qC���                                    Bxb��  �          @�{@�{��
=?�=qAZ{C�p�@�{���þǮ�9��C���                                    Bxb���  �          A
=q@�����(�@C�
A��C�e@�����{?(��@��C�P�                                    Bxb��h  �          A��@�G���  @K�A��C��=@�G�����>���@
=C��                                    Bxb��  �          A  @�p���  @FffA�(�C�#�@�p���Q�>�(�@%�C�Z�                                    Bxb�Ҵ  �          A��@Ϯ��
=@vffAə�C�E@Ϯ���
?�
=Ap�C�T{                                    Bxb��Z  
�          Aff@�p�����@��A�ffC�޸@�p�����@z�AN�RC�q�                                    Bxb��   �          A	�@�Q��]p�@�Q�B ��C���@�Q���{@c�
A�p�C��=                                    Bxb���  �          A\)@�{����@�Q�Aߙ�C���@�{����?�\)A=p�C�+�                                    Bxb�L  �          A
=@ƸR�x��@�p�B�C���@ƸR����@a�A��HC�:�                                    Bxb��  �          Aff@����_\)@��B&��C�4{@������@���A�{C�N                                    Bxb�*�  �          A��@��Ϳ��HA ��Bg��C��@�����
=@ٙ�B6�
C��\                                    Bxb�9>  �          A�@�=q�0��A  Br�C��
@�=q���H@�ffBFG�C��                                    Bxb�G�  �          A�\@�=q����@�z�BR  C��3@�=q���@��BQ�C�q�                                    Bxb�V�  �          A�\@Å�Q�@��B3�\C��)@Å��(�@�z�AC�f                                    Bxb�e0  �          A�H@�Q��g
=@׮B,{C��@�Q��Å@�33A��
C���                                    Bxb�s�  �          A��@�  �o\)@ٙ�B/z�C�
=@�  ��Q�@��HA�33C��                                    Bxb��|  �          A=q@���ff@�\BH��C�\@�����\@�(�BQ�C���                                    Bxb��"  �          A�R@�z��.�R@�Q�BD��C��3@�z���z�@���B
�C�)                                    Bxb���  �          A  @�Q��E�@�
=B'��C�Q�@�Q�����@��A��HC���                                    Bxb��n  �          AG�@���8Q�@��B0\)C���@�����@�ffA�
=C��3                                    Bxb��  �          A=q@�(��8Q�@�ffB#p�C�� @�(����H@��
Aߙ�C�3                                    Bxb�˺  �          A�R@陚�333@�33B��C�� @陚��(�@��HA�\)C�xR                                    Bxb��`  �          A (�A   �#33@�p�B=qC�'�A   ���@tz�A�C��                                    Bxb��  �          A   @�{�>{@���B	�RC�u�@�{���H@w
=A��C�C�                                    Bxb���  �          A�@���<��@��B�\C�'�@�����R@��AŅC�o\                                    Bxb�R  �          A\)@陚�X��@��B33C���@陚���@w
=A�z�C�Y�                                    Bxb��  �          A ��@�ff��33@���B  C�o\@�ff��G�@H��A�\)C�p�                                    Bxb�#�  �          A!G�@�\)���@�z�BG�C�{@�\)�˅@=p�A�z�C�W
                                    Bxb�2D  �          A z�@�G����@��HB �C���@�G���\)@>{A��C��                                     Bxb�@�  �          A   @���ff@��HA�\)C�Z�@���\)@3�
A��RC���                                    Bxb�O�  �          A Q�@�Q���p�@��\A���C�~�@�Q���{@3�
A���C���                                    Bxb�^6  �          A�@���G�@��A��
C��R@���=q@{AM��C��3                                    Bxb�l�  �          A�@�
=��(�@�33A�  C���@�
=���R?���A(z�C�.                                    Bxb�{�  �          A z�@�p��~{@���A�p�C���@�p����\@\)Ag
=C�k�                                    Bxb��(  �          A"�RA���\)@�ffA�(�C�(�A����@��AH��C�N                                    Bxb���  �          A#33A ������@�Q�A�(�C�Z�A ����p�?�z�A,��C��q                                    Bxb��t  �          A"�R@�����\@��A�33C�N@����z�?�
=A�
C�{                                    Bxb��  �          A$(�A����@O\)A��C�h�A���{?#�
@fffC�W
                                    Bxb���  |          A*{A���R@C�
A�(�C�aHA����?�R@W�C�g�                                    Bxb��f  �          A*ffA�����@.�RAn=qC�b�A����R>���?�\)C���                                    Bxb��  �          A*=qAff���@2�\As33C�\Aff���>��R?�z�C�j=                                    Bxb��  �          A*ffA����(�@<(�A�ffC��
A����z�>�G�@
=C�+�                                    Bxb��X  �          A)�A
�H��=q@!G�A[�
C���A
�H�Å���:�HC�l�                                    Bxb��  7          A*=qAff��
=@!�A\(�C��=Aff����<��
=�Q�C�q�                                    Bxb��  ?          A,��A���mp�@�A�{C��=A����Q�@��AL��C��)                                    Bxb�+J  
o          A+�A�����@qG�A���C�B�A������?\A
=C�O\                                    Bxb�9�  �          A+33A����  @[�A�{C��fA����G�?��@�(�C�@                                     Bxb�H�  
�          A+33A	����@tz�A�ffC�*=A	��\?��@��C�e                                    Bxb�W<  �          A+\)A   ��G�@�ffAĸRC���A   ��?���A	��C���                                    Bxb�e�  �          A+�
A�\���H@�ffA�(�C�ffA�\�׮?�
=Az�C�8R                                    Bxb�t�  T          A+�A{����@��A�Q�C��A{��G�?�33@��C�                                    Bxb��.  
�          A+�A����@�Q�A�\)C��A���У�?�=qA\)C�f                                    Bxb���  "          A+�
A�����@�G�A��RC��
A��\?��A(�C��                                    Bxb��z  
�          A+�A�����R@s�
A���C�Q�A����33?�33@�C��3                                    Bxb��   
�          A+�
A
ff����@y��A�Q�C��A
ff��\)?��@߮C�)                                    Bxb���  T          A+�AQ����\@p  A�{C��3AQ���\)?�G�@�  C�ٚ                                    Bxb��l  T          A-��A���z�@eA���C�K�A���Q�?��@�{C��3                                    Bxb��  �          A-G�A�
���@��
A�\)C�\A�
��\)?�ffA�C��                                     Bxb��  
�          A-p�A���\)@s33A�Q�C�)A����?��@�(�C�N                                    Bxb��^  T          A.=qA���  @O\)A�ffC�*=A���
=?n{@�p�C�޸                                    Bxb�  
�          A-A����ff@G�A�\)C���A�����H?=p�@x��C�z�                                    Bxb��  
�          A+�
Aff��33@J�HA�p�C�RAff��
=?(�@P  C��                                    Bxb�$P  "          A(z�A(����@��A;�C��A(���
=�L�;���C��                                    Bxb�2�  "          A$(�A\)���?���@���C��A\)����W
=��
=C��                                     Bxb�A�  "          A"=qA����p����333C��A���z=q��
�;�C�]q                                    Bxb�PB  
�          A#�
A���
==u>���C��A��tz���"�HC��)                                    Bxb�^�  �          A$��A�\���>�
=@�C��A�\�z�H�������C��3                                    Bxb�m�  	�          A%p�AG����Ϳ   �1G�C��\AG����'
=�j=qC���                                    Bxb�|4  ?          A%Aff��=q���
��C���Aff�w
=��
=�+�
C���                                    Bxb���  
�          A$��A���G�=�Q�>�C���A������
=�,��C�                                      Bxb���  "          A%��A����(�?8Q�@�Q�C�c�A����  �����{C���                                    Bxb��&  
�          A&�RA�H���>Ǯ@Q�C�y�A�H���
�������C�0�                                    Bxb���  
�          A&�\A33��=q?z�@Mp�C�ǮA33�����G���HC�T{                                    Bxb��r  �          A&=qAQ�����=q��33C��AQ��u��W����
C��                                     Bxb��  �          A$��AG����׿z��N{C���AG���
=�7
=���RC���                                    Bxb��  T          A$��A	�����H�W
=����C�|)A	������*�H�q��C�                                      Bxb��d  T          A%G�A
=���>��R?�(�C�#�A
=���R���R�2{C�)                                    Bxb� 
  �          A#\)A����G�?�\@7�C��HA����p����@(�C��f                                    Bxb��  T          A#33A����\)��G���{C�ٚA���c�
�8����ffC�C�                                    Bxb�V            A"ffAz���{�����z�C���Az��X���H����ffC��3                                    Bxb�+�  X          A"�\Az�������S�
C�'�Az��E��z����
C�O\                                    Bxb�:�  `          A�A\)���\)�N{C���A\)�|(��%��q��C��
                                    Bxb�IH  �          A33A�H�����2ffC��RA�H�c33�z�H���C���                                   Bxb�W�  ^          A ��A33���R�Q��[
=C�k�A33�;����
��p�C��\                                   Bxb�f�  �          A"�\A
�R��G��(���qG�C�*=A
�R�8����z���33C�Ǯ                                   Bxb�u:  
(          A#\)A���(��#33�g33C��qA��1������  C�L�                                    Bxb���  �          A#�A���p��"�\�f{C��fA��&ff��(���
=C��                                    Bxb���  ~          AA
�\�_\)��Q��;\)C�fA
�\����R�\����C��3                                   Bxb��,            A=qA
=�hQ���4��C�]qA
=��H�QG���G�C��q                                    Bxb���  �          A�RA��=p��&ff��p�C�^�A���p��i����(�C�
=                                    Bxb��x  �          A\)A�
��>u?�Q�C�\)A�
��  �(��n�RC��                                     Bxb��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���  �          A\)A
=�8�ÿB�\���C�A
=�\)��(��A�C��R                                   Bxb��j  R          AffA33�Mp����
���C��qA33�33�%��Q�C���                                   Bxb��  ~          A�HA
ff�'��Vff����C���A
ff�Tz���p�����C�>�                                   Bxb��  ~          A�A33�����fff���C��A33�����H�ɅC���                                    Bxb�\  �          A33A(��333�+���G�C��A(�����i�����C��                                   Bxb�%  ^          A33A	p��9���G��`��C��=A	p������U���
=C��\                                   Bxb�3�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�BN              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�P�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�n@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�|�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��p              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb� �              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�b              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�;T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�I�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�X�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�gF              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�u�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�h              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�%�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�4Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�C               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�Q�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�`L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�n�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�}�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��"              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�-`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�J�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�YR              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�g�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�v�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��D            A"�\@�
=?޸R�
�R�k  A�ff@�
=@��������+z�B@Q�                                    Bxb���  

          A   @l��@W�����nB)p�@l��@������ffBw�                                    Bxb���  
�          AG�@9���
=�����C�E@9��@aG��G��r33BJ=q                                    Bxb��6  "          A�?��H�u�¢�C�Z�?��H@��H��R�z��B��)                                    Bxb���  �          A�\?�녿#�
��W
C�o\?��@k���(�B�Ǯ                                    Bxb�΂  
�          A��@p��c�
�
=33C�\@p�@U�����)Ba                                    Bxb��(  
�          A�\@{�������C��R@{@{�\)G�B3�R                                    Bxb���  
�          A��?���Q��ffp�C��)?�@\)�(��BVG�                                    Bxb��t  
Z          A�H?333?   ���©� B  ?333@�
=� Q��m
=B�=q                                    Bxb�	  
�          A
=?�\)�W
=��
 =qC�=q?�\)@`���	p���B�G�                                    Bxb��  T          A��?�  ��\�z�G�C��q?�  @Q�����B[33                                    Bxb�&f  "          A
=@�R�����H  C��@�R@Fff�33��BMp�                                    Bxb�5  "          A��@J=q�ٙ����C��H@J=q@%��
�\�B                                    Bxb�C�  @          A��@*=q���H�
�\�C��@*=q@:=q��
��B>ff                                    Bxb�RX            A�@2�\�\�ff�fC��@2�\@p�����o�\BU��                                    Bxb�`�  
�          A��@@  >�
=��Rz�A   @@  @����
=�\=qB`�H                                    Bxb�o�  �          Az�@QG��^�R�����C���@QG�@R�\���r\)B4�\                                    Bxb�~J  T          AG�@\(����
�	�\)C�Z�@\(�@Q���H�~ffB
\)                                    Bxb���            A�@@  �.�R��z��q�\C��@@  ?^�R���H8RA�p�                                    Bxb���  �          Az�?}p���
=��(��>�C��f?}p��ff��L�C�s3                                    Bxb��<  �          A�H?\(�����\�O�C�Q�?\(���z����
C��)                                    Bxb���  "          A�>�\)��Q�������C��>�\)��
=���u�C��\                                    Bxb�ǈ  �          A�?�����H��(��9��C�=q?������z���C���                                    Bxb��.  �          A�@Mp����
��  �9(�C�XR@Mp���p���R�C�*=                                    Bxb���  T          A=q@�����p���ff�9�C���@��ÿ��
���{
=C�4{                                    Bxb��z  
�          A{?�G�������z��M�RC�H�?�G���\)�	
=C��)                                    Bxb�   
�          A  @(���p�����EG�C��@(������33C�XR                                    Bxb��  T          A�@ff��  �أ��F��C�3@ff���
��H(�C�u�                                    Bxb�l  T          A��@   ����=q�a��C�!H@   �\)����3C�7
                                    Bxb�.  �          Aff@I�����R�Ϯ�/�C��=@I����\�(���C��3                                    Bxb�<�  T          A��@%���H��Q��!G�C��
@%�E����3C��                                    Bxb�K^  "          A�?У��߮�����C��?У��mp�� Q��|��C�ٚ                                    Bxb�Z  
�          Az�?���������(��D{C��q?��ÿ�Q��Q�.C�xR                                    Bxb�h�  �          A�
?���������\)�Y��C��
?��ÿ�G��\)C�u�                                    Bxb�wP  "          A\)>\��
=��R�R��C���>\���
��R   C��                                    Bxb���  �          Az�@{��p��ȣ��%�C�,�@{�AG��
=qaHC��f                                    Bxb���  �          Ap�@Q���\)��33���C��3@Q��g
=�
=�l��C��                                    Bxb��B  �          A��@P����Q������{C�t{@P���k��{�kz�C��                                    Bxb���  �          AG�@ ����\��ff�(�C�l�@ �����R�G��r��C��H                                    Bxb���  �          A\)?�{��=q������
C��)?�{�`��� ���~�C��                                    Bxb��4  
�          A
=@QG����
��z��
=C��3@QG��n�R��\)�eG�C��R                                    Bxb���  
�          A�@^�R�ᙚ����

=C�%@^�R�p��� ���f
=C�ff                                    Bxb��  
(          A��@����{��ff��Q�C�@���������G�HC�]q                                    Bxb��&  
�          A{@����z���ff����C�  @�����H��  �C��C��q                                    Bxb�	�  
Z          A�@]p�� ���vff���C��H@]p����
��(��@�C�h�                                    Bxb�r  �          A
=@K��Q��p����  C���@K���������?�C��f                                    Bxb�'  T          A�@S�
�������R�ffC�� @S�
�n{�G��ip�C���                                    Bxb�5�  "          A�@������\)�C�>�@���������H�ip�C��=                                    Bxb�Dd  �          A�H>�=q��H�Dz���(�C��>�=q��=q���
�=��C�K�                                    Bxb�S
  
�          Aff@�p����H�C33���HC�� @�p��*�H��=q�"�C��                                    Bxb�a�  �          A!G�@�z��n�R����8�
C���@�z�=#�
��R�X�>Ǯ                                    Bxb�pV  
�          A$��@�{�tz���\)�:�\C��)@�{=L���ff�Z��?�\                                    Bxb�~�  
�          A*{@�(�����ff�233C�@�(��   �
ff�ZffC��                                    Bxb���  T          A'\)@�33��p��޸R�&�HC��
@�33�   ����K
=C��
                                    Bxb��H  T          A)�@�z��tz��ٙ�� �C��=@�zᾅ������>
=C���                                    Bxb���  
�          A)�@��
��{���*  C��{@��
�(����R�C��                                    Bxb���  "          A*�R@������������C�}q@��Ϳ�����I�C�q                                    Bxb��:            A&�R@�=q��ff���R��G�C���@�=q�?\)��ff�?{C�U�                                    Bxb���  ,          A)@�=q��ff�x����p�C�XR@�=q�mp�����33C��                                    Bxb��  �          A0z�@���ۅ�j=q���C�XR@���������33C�G�                                    Bxb��,  T          A4  A��ə�@g
=A�
=C���A���  >��R?˅C��)                                    Bxb��  T          A0��Ap���(�@Q�A2ffC�:�Ap����H���
����C��\                                    Bxb�x  �          A1A���Q�@|��A��C�ٚA��ָR?xQ�@���C�5�                                    Bxb�   T          A0��A{��\)@�(�A�Q�C�g�A{��  ?˅A(�C�3                                    Bxb�.�  
�          A1G�A(����@�(�A�z�C�A(���33?\@�(�C���                                    Bxb�=j  
�          A1��AQ���p�@Q�AG
=C�Y�AQ���33�(��J�HC�j=                                    Bxb�L  I          A/
=@������@�Q�B��C��
@������@7�Aw�C���                                    Bxb�Z�  
�          A,��@ڏ\��(�@�p�B{C��H@ڏ\��ff@#33AZ�RC�XR                                    Bxb�i\  �          A-G�@��H��@���A�C�"�@��H��\@�\AC�C��                                    Bxb�x  
�          A.ffA����
=@Y��A�p�C�:�A�����H>#�
?^�RC�e                                    Bxb���  �          A/
=Ap����H@AG�A���C�` Ap���׾k����HC��{                                    Bxb��N  �          A/�
A���
=?���A��C���A���{��Q��(�C��
                                    Bxb���  �          A1p�A�\���@<��A�=qC�p�A�\�Ӆ��Q�   C��=                                    Bxb���  
Z          A5G�A ���1G�@�Q�B!��C�y�A ������@�p�A܏\C�R                                    