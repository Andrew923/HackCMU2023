CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230412000000_e20230412235959_p20230413021737_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-13T02:17:37.482Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-12T00:00:00.000Z   time_coverage_end         2023-04-12T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxw,1@  "          @��?�@���@z�A�\)B�?�@\(�@�Q�BW(�B�Q�                                    Bxw,?�  �          @�ff?(�@�z�@(�A�(�B��?(�@W�@��B[Q�B��\                                    Bxw,N�  "          @�G�>�
=@�Q�@�HA�\)B�Q�>�
=@_\)@���BY�B�Q�                                    Bxw,]2  �          @���=�Q�@��\@(�A��B�(�=�Q�@j�H@��BP�HB�33                                    Bxw,k�  T          @�{�k�@�z�?�ffA��B��\�k�@z�H@�{BB  B��\                                    Bxw,z~  
�          @Å=#�
@�G�?���A��B�L�=#�
@~{@�z�BEB���                                    Bxw,�$  
�          @�ff>8Q�@�33@�\A�B�p�>8Q�@~�R@�  BG��B�Ǯ                                    Bxw,��  T          @�Q�=��
@�@�\A�Q�B�p�=��
@���@�p�BDG�B�                                    Bxw,�p  T          @�{>aG�@�ff?�(�Ax(�B�>aG�@�ff@���B9  B�p�                                    Bxw,�  T          @���\)@���?�A/
=B�=q��\)@�33@�33B&B���                                    Bxw,ü  "          @���>��@��\?���A^�RB��>��@���@�\)B2z�B��                                    Bxw,�b  �          @���=��
@�\)?�33AR=qB��=��
@�@�Q�B/p�B�                                      Bxw,�  T          @Ǯ����@\?�{AJ=qB��f����@���@���B-\)B��=                                    Bxw,�  �          @ƸR=�@�p�?0��@˅B��3=�@�ff@o\)B33B�#�                                    Bxw,�T  �          @ʏ\=��
@�Q�?#�
@��
B��{=��
@��@p  B(�B�B�                                    Bxw-�  T          @�>k�@�33?}p�A=qB��)>k�@��R@��HB�B��3                                    Bxw-�  T          @�  ��@���?xQ�A
ffB�(���@�Q�@�33B33B�Ǯ                                    Bxw-*F  "          @�  �B�\@�\)>��@��
B�ff�B�\@��\@l��B
��B��                                    Bxw-8�  
�          @��=�@��H?\(�@��RB��f=�@���@~{BQ�B�\)                                    Bxw-G�  
�          @�33<#�
@ə�?:�H@�33B��f<#�
@���@uB�
B��H                                    Bxw-V8  �          @ə�>��@�\)?=p�@أ�B�=q>��@��@s�
BQ�B�
=                                    Bxw-d�  
�          @�G���@�p�?���A%G�B�z��@��@��B#�\B�8R                                    Bxw-s�  �          @�논#�
@ȣ�?�\@��B�.�#�
@���@hQ�B�B�33                                    Bxw-�*  �          @�Q�?333@�33�}p��{B��H?333@�Q�@  A��B��                                    Bxw-��  "          @�(�?   @��R�@  ��B�Q�?   @�Q�@Q�A�ffB��=                                    Bxw-�v  "          @ȣ׼��
@Ǯ>�(�@z=qB�\)���
@��@a�B	�B�k�                                    Bxw-�  
�          @�G����
@�  ?��@�=qB�ff���
@��@i��B=qB��3                                    Bxw-��  T          @��;�\)@��
>�Q�@P  B�\��\)@��@b�\BB�\                                    Bxw-�h  T          @����Q�@�z�>W
=?�z�B�ff��Q�@���@Z�HBG�B��{                                    Bxw-�  
�          @��H��@ə�>��R@2�\B��=��@���@]p�Bz�B�aH                                    Bxw-�  �          @�z�?(��@��?8Q�@�{B�.?(��@�{@eB�\B�                                    Bxw-�Z  �          @�z�@/\)@���?@  @���B|p�@/\)@���@XQ�Bz�Be�                                    Bxw.   
Z          @��@n�R@��?fffA�BP��@n�R@l(�@L��A�33B2�                                    Bxw.�  "          @�p�@p  @�=q?5@���BPff@p  @s33@C33A�  B5ff                                    Bxw.#L  
�          @��@n{@�33?Q�@�BR
=@n{@qG�@I��A�=qB5��                                    Bxw.1�  
�          @�@q�@�  ?�z�A,��BM��@q�@`��@Y��BffB+�                                    Bxw.@�  	�          @ƸR@q�@�G�?���A�BO{@q�@fff@VffBB.ff                                    Bxw.O>  
�          @Ǯ@`��@��?#�
@�p�B]��@`��@��\@HQ�A�p�BE=q                                    Bxw.]�  	�          @�(�@@��@�=q>�  @
=Br=q@@��@�
=@9��A�\B`\)                                    Bxw.l�  
�          @��
@Q�@�(���G���33Bez�@Q�@���@p�A���B[{                                    Bxw.{0  T          @�z�@N{@�
=<��
>aG�Bi(�@N{@�  @)��Ạ�BYff                                    Bxw.��  T          @�p�@e�@��׽L�;�BZ�@e�@��@�RA�=qBJ��                                    Bxw.�|  T          @��@(Q�@��=u?��B��H@(Q�@�
=@3�
A�=qBs�                                    Bxw.�"  
�          @�p�?���@��
>�(�@�Q�B�  ?���@�33@Tz�Bp�B�G�                                    Bxw.��  �          @�ff?�
=@��
>��
@@��B��{?�
=@��@N�RA��
B�{                                    Bxw.�n  "          @�{@G�@�=q>��H@���B���@G�@�G�@VffB(�B�\)                                    Bxw.�  
�          @�  @��@��?�@��B��
@��@�ff@U�B33B|\)                                    Bxw.�  T          @��
@��@�(�?B�\@�B�Q�@��@��@^�RB�B�\                                    Bxw.�`  N          @\@Q�@�=q��\)�+�B��q@Q�@�
=@#33AƸRB�                                    Bxw.�  
�          @���@\)@��R>�  @=qB���@\)@�33@=p�A�Bv33                                    Bxw/�  �          @�Q�@�
@�\)>B�\?���B�33@�
@���@:=qA�=qB~�                                    Bxw/R  �          @��\@�  @o\)�8Q���ffB,p�@�  @fff?�  AQ�B'��                                    Bxw/*�  T          @�=q>.{@��>�{@��B�u�>.{@`  @=qB
33B��                                     Bxw/9�  "          @�{�"�\@���?�ffA~�HB�Ǯ�"�\@XQ�@o\)B&  B���                                    Bxw/HD  "          @������R@��R?ǮAy�B��
���R@n�R@|��B,=qB�#�                                    Bxw/V�  
�          @����*�H@�ff?�z�A;
=B��*�H@mp�@^�RB�B���                                    Bxw/e�  
�          @��H�R�\@�ff?�ffA�Q�B��R�\@=p�@q�B"�C

                                    Bxw/t6  	�          @���Z=q@�(�@�A�33C���Z=q@��@��
B2��C�)                                    Bxw/��  
�          @�z��=q@�=q?ǮA��HB۸R��=q@g�@xQ�B.��B��                                    Bxw/��  T          @�(��
=q@�{?�{A���B�(��
=q@^�R@w
=B-{B�                                    Bxw/�(  
Z          @����@�(�@	��A�{B�����@J=q@�G�BAB��3                                    Bxw/��  �          @�  �z�@��@1G�A�B��z�@%�@�BZ�C ��                                    Bxw/�t  �          @����@���@\)A��HB��
��@.{@�p�BPp�B��)                                    Bxw/�  �          @��H�{@�p�@$z�A��
B�W
�{@#�
@�BR\)C�3                                    Bxw/��  �          @����
=@��@:=qA���B�W
�
=@�@�G�B]Q�C
��                                    Bxw/�f  
�          @����\)@��@5A�33B���\)@�@�33Bc�C ��                                    Bxw/�  �          @���<��@���@33A�B��H�<��@;�@�G�B0�C!H                                    Bxw0�  �          @�z��"�\@�33?�A_33B�
=�"�\@n�R@p��B
=B���                                    Bxw0X  
�          @�(���@���?���A�z�B�q��@\��@��HB0\)B�u�                                    Bxw0#�  �          @��,��@�{@�\A��\B��,��@.{@�{B>{C�=                                    Bxw02�  �          @�Q���@���@�
A�{B�L���@:�H@���B<C ��                                    Bxw0AJ  
�          @�z��C33@mp�@�
A�G�Cn�C33@@uB8\)C��                                    Bxw0O�  �          @��
�\)@<(�@z�A�G�C���\)?��@P  B{C��                                    Bxw0^�  T          @��R�s33@>�R?��HA�
=C޸�s33?У�@K�BQ�C�
                                    Bxw0m<  T          @���x��@7�@   A�C���x��?�G�@J=qB\)C�                                     Bxw0{�  	�          @��R�c�
@u�?�ffA�  C���c�
@$z�@N�RBz�C.                                    Bxw0��  
�          @��R�QG�@���?��A���C��QG�@,(�@Z=qBQ�C�=                                    Bxw0�.  "          @�  �X��@p��@�
A��C��X��@��@h��B'33CL�                                    Bxw0��  
(          @�=q�i��@l(�@ ��A�\)C���i��@{@c�
B=qC�f                                    Bxw0�z  
�          @�z����@W�?���A�G�C�R���@G�@P��B�C��                                    Bxw0�   
�          @����ff@@��?�  A�=qC�f��ff?�\@@  B ffCY�                                    Bxw0��  �          @�(�����@,��@=qAˮC����?�@Z�HB��C$�3                                    Bxw0�l  T          @�G��h��@�@B�\B�HC��h��?   @qG�B6�C,!H                                    Bxw0�  �          @����;�?�z�@z=qBI(�CW
�;��0��@�33BV�CA:�                                    Bxw0��  T          @�{�?
=@���Bxp�C%�q����H@�Q�B_=qCX33                                    Bxw1^  
�          @�  ���@=q?�G�Aa�C\)���?�p�@�A��HC!h�                                    Bxw1  
�          @����33@ff?c�
A��C����33?��?�ffA�
=C#��                                    Bxw1+�  �          @�p����\@a녾.{����C(����\@J=q?���A�33C5�                                    Bxw1:P  T          @�z���
=@\)?Y��A��C�)��
=?��?�A��\C!�                                    Bxw1H�  T          @����=q=���@<��B�
C2�)��=q����@(Q�A��CFaH                                    Bxw1W�  �          @�33�j=q�:�H@EB��C?:��j=q���@ffA��
CRT{                                    Bxw1fB  
�          @���|(����
@;�B�\C4G��|(���Q�@#33A�ffCH)                                    Bxw1t�  "          @��\��{?��@)��A�  C%����{��Q�@5�B�RC8�3                                    Bxw1��  �          @�Q��y���+�@@  B��C=���y����\@�\A�z�CO��                                    Bxw1�4  N          @�{�g
=�z�H@@��BQ�CC5��g
=�z�@
=qA�p�CT�R                                    Bxw1��  Z          @�
=�Tz�^�R@c33B7CB��Tz���R@+�B�\CX                                    Bxw1��  
�          @�G��i��?У�@\)A��C��i��>aG�@>{B33C0�                                    Bxw1�&  
�          @�(����?�  @#�
A��C&Q������Q�@.�RBC9�                                    Bxw1��  �          @�ff�9��?�?���A�Q�Cs3�9��?J=q@#33B!\)C$��                                    Bxw1�r  
�          @�  �@��?(�@أ�B�{�@vff@5�BffB��                                    Bxw1�  �          @�\)��z�@�(�>\)?�ffB޸R��z�@�ff@�RA��B���                                    Bxw1��  
�          @��ÿ�\@�
=�#�
�L��Bي=��\@�=q@ ��A�
=B�ff                                    Bxw2d  "          @�=q�Mp����
@G�A�{CQ�Mp��*=q?���As\)C[��                                    Bxw2
  
�          @��\�{�>{@(�A�\Ci8R�{�j=q>��H@У�Cn�q                                    Bxw2$�  
�          @�33�s33�
�H@!�A��CQ�q�s33�G�?�
=AW\)C[h�                                    Bxw23V  
Z          @�����ÿ�G�@'�A��C@���������?���A�\)CMY�                                    Bxw2A�  
�          @�ff���׿0��@��A���C<J=���׿�G�?�G�A���CH8R                                    Bxw2P�  
�          @�\)���H�Q�@
=A��HCL޸���H�8Q�?Tz�A
�HCT{                                    Bxw2_H  "          @�  ������?���A���C@  �����{?���AJ�HCHxR                                    Bxw2m�  
�          @�\)��{�!G�?��A�
=C:�f��{��=q?���A?\)CBn                                    Bxw2|�  "          @��������8Q�?�G�A~ffC5������aG�?��RAP  C=k�                                    Bxw2�:  T          @�p����R=�Q�?�(�A{�C3���R��R?�=qAc�C:��                                    Bxw2��  
�          @����{>��?ٙ�A���C.����{����?�(�A��\C8h�                                    Bxw2��  
(          @�Q���Q�>�Q�?��RA}G�C0!H��Q����?�p�A{\)C8\)                                    Bxw2�,  
Z          @�  ���R����?��A���C8aH���R��=q?�z�AD��C?�q                                    Bxw2��  �          @��\��=q>�p�?��\AUC/�R��=q��=q?��AZ{C6�3                                    Bxw2�x  �          @�����?Q�?�A�{C+.���þ#�
@ ��A��\C5��                                    Bxw2�  �          @����=q?�?�ffA�\)C!�R��=q?=p�@
=qA��RC+��                                    Bxw2��  "          @����=q?
=?��A�{C-����=q���?�(�A�\)C6                                    Bxw3 j  
�          @������?\)?�ffA��C-�3���þ�Q�?���A�=qC7�                                    Bxw3  "          @�Q�����>B�\?�\)A�
=C2
=���׿��?\At��C:=q                                    Bxw3�  T          @�  ����?
=?�AeG�C-����þ��?��
Aw\)C5�
                                    Bxw3,\  �          @�=q���@,(�?333@�  C�{���@�?��A�
=C{                                    Bxw3;  �          @�������?�?L��A	p�C!�3����?��?�  A�\)C'��                                    Bxw3I�  T          @�z���p�@ ��?J=qA33C�q��p�?�?��A�Q�C}q                                    Bxw3XN  �          @�����@.{?�
=AI��C  ����?�@�A˅C޸                                    Bxw3f�  T          @����33@{?��A��HCn��33?�p�@��A�G�C%Ǯ                                    Bxw3u�  
�          @������?�Q�@�RAظRC�\����>���@?\)B��C/�f                                    Bxw3�@  
�          @�33��{?.{@?\)B �HC+����{�L��@=p�A���C=��                                    Bxw3��  T          @��R���?�z�@333A��HC#+������\)@HQ�B�C4�)                                    Bxw3��  �          @�\)�q�@w�?�p�A�  CJ=�q�@'
=@VffBp�Cff                                    Bxw3�2  
�          @�  ��p�@>�R@�RA�=qCaH��p�?�(�@eB=qC ��                                    Bxw3��  T          @���qG�@i��@!G�A�p�C��qG�@�\@{�B*G�C�\                                    Bxw3�~  T          @���o\)@p  @
=A��C�f�o\)@p�@uB%CaH                                    Bxw3�$  "          @���W
=@��
@(�A�G�CB��W
=@'
=@w
=B(��C(�                                    Bxw3��  "          @����#�
@��R?�\)A�{B��#�
@Dz�@o\)B,\)C޸                                    Bxw3�p  
�          @����:=q@qG�@ffA�Q�B�\)�:=q@ff@g�B0  C�                                    Bxw4  
�          @�  ��=q@33?��RAo
=C����=q?��@��A�C%+�                                    Bxw4�  �          @�ff���?aG���G���33C*�\���?Tz�>��
@W
=C+�                                    Bxw4%b  �          @�����\)?�?�@�\)C.n��\)>k�?8Q�@�{C1��                                    Bxw44  "          @������R?\)?�\@�Q�C.����R>��?5@���C1:�                                    Bxw4B�  �          @�����(�?�G�?B�\@��C)aH��(�?
=q?��A>{C.=q                                    Bxw4QT  
�          @������\?��\?z�@���C&�f���\?Y��?���A8  C*��                                    Bxw4_�  T          @�����G�?���>�\)@;�C#����G�?�G�?z�HA"�RC&�H                                    Bxw4n�  T          @�Q���=�?5@�  C2� ���k�?0��@�\)C6aH                                    Bxw4}F  "          @�����z�?�ff>�@�(�C#�R��z�?�33?��A;
=C'�                                    Bxw4��  �          @�����?�G�?+�@�\)C(�����?
=?��A8(�C-�                                     Bxw4��  T          @�(���ff?n{?aG�A�RC)ٚ��ff>�
=?��HAPz�C/^�                                    Bxw4�8  	�          @����\)?�  ?���A<��C �)��\)?�ff?��
A��RC(�                                    Bxw4��  �          @�����\)@�\?xQ�A)�C����\)?�\)?�ffA�ffC#��                                    Bxw4Ƅ  T          @���{@(Q�>�p�@�z�Ck���{@
=q?��A�ffC�                                    Bxw4�*  
Z          @�����{@\)?G�A
�RC����{?�{?�=qA���CE                                    Bxw4��  
�          @��H��G�@HQ�8Q��Q�C�
��G�@7�?��
A^�\C=q                                    Bxw4�v  �          @�G��s33@H�þ��H��ffCk��s33@AG�?�  A:�\C�=                                    Bxw5  "          @�
=�  @{��?\)��RB�  @���\(���B�3                                    Bxw5�  !          @����(�@�G��/\)��\)B�{�(�@�녾����RB�R                                    Bxw5h  �          @�=q�G�@��\)�ԏ\B�ff�G�@�=q�B�\��p�B�L�                                    Bxw5-  "          @�=q��@���p���G�B�aH��@��>\)?��HB�33                                    Bxw5;�  T          @�=q��@�������\)B�q��@��
�\)��B��                                    Bxw5JZ  �          @�=q���@���p���\)BӅ���@�(�>�=q@0  B��f                                    Bxw5Y   �          @�z���
@��R�
=q���\B�G���
@�>W
=@��B�k�                                    Bxw5g�  "          @�����@�z���H�͙�B�B����@�Q�#�
��B�\                                    Bxw5vL  "          @�G��E�@�\)��ff��p�B�#��E�@��\>��
@S33B���                                    Bxw5��  T          @��
�`  @z�H��33���C���`  @�33=��
?Q�C �\                                    Bxw5��  T          @�ff�\��@�  �Ǯ�\)C
=�\��@�  ?�@�=qB��                                    Bxw5�>  
�          @�z��X��@�Q��G��x��C z��X��@�\)?z�@��HB��                                    Bxw5��  �          @�(��l(�@�녿Y���G�C8R�l(�@�Q�?���A2=qC��                                    Bxw5��  �          @�Q��&ff@s�
@z�A��B����&ff@ff@r�\B<��C	�
                                    Bxw5�0  �          @�z��>�R@�33?��A�p�B���>�R@6ff@\(�BG�CB�                                    Bxw5��  
�          @�=q�L(�@�ff?�G�A&�RB�8R�L(�@aG�@9��A�G�C&f                                    Bxw5�|  "          @����Y��@�G�?^�RAQ�C h��Y��@\(�@,��A�C�f                                    Bxw5�"  "          @����N{@�{=�?��RB��H�N{@z�H@ffA��Ck�                                    Bxw6�  
�          @�p��,��@��R>�p�@o\)B��,��@���@#�
A֣�B�                                    Bxw6n  �          @����`  @��?�
=A=p�C xR�`  @Z=q@A�A�(�C�q                                    Bxw6&  �          @���\��@�(�>�Q�@eB�W
�\��@~�R@��AÙ�C�f                                    Bxw64�  "          @�  �W�@���#�
��=qB���W�@�
=@�\A��\B�{                                    Bxw6C`  �          @�Q��\��@���
=��ffB����\��@��
?��A}p�B���                                    Bxw6R  
�          @�\)�S�
@���}p����B�z��S�
@�G�?��
AF{B�=q                                    Bxw6`�  �          @�33�G
=@��
�u��B���G
=@�G�?�ffAM��B��                                    Bxw6oR  
�          @����'
=@�{���H�iB��'
=@�=q?c�
A�B�p�                                    Bxw6}�  "          @�33�HQ�@�G��G���G�B�#��HQ�@�33����ffB��\                                    Bxw6��  
�          @���B�\@hQ��$z���C�q�B�\@�z�0����z�B�u�                                    Bxw6�D  
�          @�(��AG�@`���(���ffC���AG�@��
������\B��                                     Bxw6��  T          @�  �0��@C33��=q��(�C��0��@c33�����p��B��R                                    Bxw6��  �          @����@��@8�ÿ�
=���HC8R�@��@N{    �L��C
                                    Bxw6�6  �          @����1G�@Mp�>��
@��CǮ�1G�@/\)?��HA�33CQ�                                    Bxw6��  �          @���]p�@"�\?8Q�A(�C���]p�?�(�?�G�A��C\)                                    Bxw6�  T          @����Vff@!�>��@��
C�3�Vff@?��RA��RC
=                                    Bxw6�(  �          @����]p�?�ff?��\A��Cff�]p�?(�?��HA�  C*�                                    Bxw7�  	`          @{��H�þaG�@G�B�C8  �H�ÿ�Q�?�Q�A��CH�                                    Bxw7t  "          @���b�\?aG�@	��A�\C&�b�\���@z�B  C6u�                                    Bxw7  �          @�������@4z�?ٙ�A�(�Cff����?��@0��A�Cc�                                    Bxw7-�  
�          @�����\@8Q�?�A���C5����\?��
@:=qA�  C�R                                    Bxw7<f  
�          @������@G�?�33A[�
C�����@��@'�A�p�CY�                                    Bxw7K  "          @��H����@_\)?���A=��CQ�����@'�@&ffAӮC��                                    Bxw7Y�  "          @��H�QG�@��H�����733B����QG�@�z�?fffA�B�ff                                    Bxw7hX  �          @���Dz�@��H��  �{\)B��=�Dz�@��?
=q@��B��                                    Bxw7v�  �          @����P  @���0����p�B��q�P  @���?��AX��B��                                     Bxw7��  
�          @�Q��5@��\������B���5@y��?�ffApz�B�\                                    Bxw7�J  T          @��R�9��@�Q�.{��\)B��=�9��@�z�?��RA[�
B�(�                                    Bxw7��  T          @��R�fff@�=q��
=��=qCǮ�fff@��\?��RAtQ�Ch�                                    Bxw7��  T          @�\)�_\)@��þ�G���p�B�33�_\)@���?ǮA|  C33                                    Bxw7�<  T          @����U@�p���  �{B�#��U@�=q?��A�33B�k�                                    Bxw7��  �          @����^�R@�33��G���\)B�8R�^�R@�ff?��A�  C�f                                    Bxw7݈  �          @��
�a�@�Q쿏\)�0��C 
=�a�@���?n{A�HB��\                                    Bxw7�.  �          @���L(�@��R��Q��=B�=q�L(�@���?p��Ap�B���                                    Bxw7��  T          @��
��@��������RB��f��@��>��
@I��B��H                                    Bxw8	z  T          @��
�AG�@��Ϳ^�R��B�{�AG�@�=q?��HAF�HB���                                    Bxw8   �          @������@�����{�Z�\B�q���@���?n{A�B��H                                    Bxw8&�  
(          @�33�)��@�녿����=p�B�aH�)��@��H?�=qA*=qB�#�                                    Bxw85l  T          @��H�E�@��
�aG��
�HB����E�@���?��
AK33B��{                                    Bxw8D  �          @�ff�J�H@�{�Y���(�B�Q��J�H@��\?�=qAO�
B�u�                                    Bxw8R�  "          @��
�i��@�(�����ffB���i��@�z�?���Ar{C :�                                    Bxw8a^  
�          @�Q��g
=@�  �z�H�(�B��3�g
=@�{?��RA6�HB�W
                                    Bxw8p  
�          @ə����
@�z῁G��  C�H���
@�(�?�ffA{C�                                    Bxw8~�  �          @�  ����@���z�H��C�����@�(�?��A z�C(�                                    Bxw8�P  
�          @�z����R@��Ϳ�p��1p�C(����R@�\)?Y��@��C�f                                    Bxw8��  "          @����w�@�����ff�aC ޸�w�@�Q�?�R@��
B�ff                                    Bxw8��  T          @�G��o\)@�ff��=q��C z��o\)@�G�>��
@:�HB�(�                                    Bxw8�B  "          @ʏ\�g�@�Q���\��z�B���g�@�>#�
?�B���                                    Bxw8��  �          @�{�]p�@����
=q���B�=q�]p�@�(�    <#�
B�                                      Bxw8֎  �          @�{�o\)@���33��z�C�\�o\)@�{<�>uB�W
                                    Bxw8�4  T          @��
�vff@�������RC@ �vff@�������z�C
                                    Bxw8��  "          @����W
=@���   �ď\B��)�W
=@��R����{B�8R                                    Bxw9�  
�          @�G��5�@����5��ffB�{�5�@�G��0����=qB�=q                                    Bxw9&  �          @�ff��@�=q�L(���B�\��@�\)�}p��p�Bݣ�                                    Bxw9�  
�          @�
==�\)@������\)B�u�=�\)@��R>���@Mp�B��\                                    Bxw9.r  "          @�Q�B�\@��H�-p�����B�\�B�\@�\)�u�G�B���                                    Bxw9=  
Z          @�Q�>���@��\�L(�� (�B��>���@�ff�J=q����B���                                    Bxw9K�  
�          @�  >W
=@���S�
�Q�B�.>W
=@���s33�p�B��                                    Bxw9Zd  T          @�
=?!G�@��\�A����B���?!G�@�(��&ff��=qB��)                                    Bxw9i
  �          @�ff>��@�ff�8Q���B��3>��@�p������G�B��                                    Bxw9w�  "          @��R�+�@�ff�AG���G�BÀ �+�@�Q�5��ffB�\)                                    Bxw9�V  
Z          @\�7�@`�������&Q�CE�7�@���{����B�(�                                    Bxw9��  T          @�{��@_\)�~�R�A�B�z��@���(���=qB�p�                                    Bxw9��  "          @��
?�\)@z�H�{��(��B�?�\)@�{��(���p�B�ff                                    Bxw9�H  �          @��H?�Q�@s33�}p��+�B{?�Q�@��H��
��p�B�G�                                    Bxw9��  T          @�=q?�33@l������1�B{=q?�33@�G������\B���                                    Bxw9ϔ  T          @�z�?�33@h����p��B�B��
?�33@�(��#33���HB�p�                                    Bxw9�:  
�          @���?u@e����R�H��B��?u@��H�&ff��p�B��R                                    Bxw9��  
�          @�����@`�����
�?  B�3����@���#�
��\)B�Ǯ                                    Bxw9��  T          @���    @b�\��z��R�B��    @��
�333��
=B��                                    Bxw:
,  T          @�=q?���@��\�}p��$�RB�(�?���@�=q��{��B�aH                                    Bxw:�  "          @�
=@#�
@xQ��o\)�(�Bb{@#�
@�녿����B|�                                    Bxw:'x  T          @�z�@H��@����W��G�BP��@H��@�녿�Q��X��Bh��                                    Bxw:6  
�          @ȣ�@S33@w��g
=��BFp�@S33@�  ��p����Bb�                                    Bxw:D�  �          @�{@E�@[���\)�0��B@\)@E�@�ff�-p���\)Bhp�                                    Bxw:Sj  �          @�\)@<��@H�����7{B;33@<��@���1���(�Bf�                                    Bxw:b  
�          @�{@'�@]p�����4ffBSff@'�@�p��%��p�Bw�
                                    Bxw:p�  
�          @��
@!G�@Q���Q��>
=BQ�H@!G�@�=q�3�
����By��                                    Bxw:\  T          @��
@�\@QG����H�G�HBh33@�\@��H�8����z�B�G�                                    Bxw:�  
�          @��R?�@�(���  �6Q�B���?�@����R��z�B�(�                                    Bxw:��  
�          @��R?5@~{��(��=�B��{?5@�z���H��=qB��f                                    Bxw:�N  T          @���?��@j�H��(��I�RB��?��@�
=�1���Q�B�k�                                    Bxw:��  �          @�33?@  @u���33�F��B���?@  @�33�+���\)B�#�                                    Bxw:Ț  �          @�\)�Y��@��H�q��33B��)�Y��@�ff����ep�B�8R                                    Bxw:�@  T          @�=q?�@���ff�'{B�\)?�@�
=�   ��p�B�{                                    Bxw:��  �          @�Q�>��H@�(��w��Q�B�k�>��H@�Q��\)�p(�B���                                    Bxw:�  �          @�G�>.{@�z��|(��33B�>.{@������w
=B�Ǯ                                    Bxw;2  T          @�z�?�@�p��i���p�B�k�?�@�{����=�B�W
                                    Bxw;�  �          @�G��.{@����c�
�
�\B��.{@�(���p��5B�(�                                    Bxw; ~  "          @�녿z�@����<(���{B��H�z�@�Q��\���
B��\                                    Bxw;/$  �          @�  �k�@��
��=q���B�Ǯ�k�@��
?!G�@��B�\                                    Bxw;=�  T          @��
�h��@��ÿ޸R�~=qB���h��@�\)?G�@�G�BĔ{                                    Bxw;Lp  T          @�(����@��\�&ff���B�Ǯ���@�{�Ǯ�k�B�8R                                    Bxw;[  �          @�  �+�@����\)��33B���+�@�33���
�>�RB��                                    Bxw;i�  "          @���=p�@�\)��(���=qB�p��=p�@�Q�>�p�@`  B�                                    Bxw;xb  
�          @�  �33@�G���ff���\B�\)�33@�=q>��@\)B�ff                                    Bxw;�  �          @��
���
@��H�"�\���
B�ff���
@��
�u��\B�\)                                    Bxw;��  
�          @ʏ\���
@��.�R���B�ff���
@ə������/\)B�                                    Bxw;�T  
�          @������@�ff�S33�p�Bԙ�����@�33��{�'�
B���                                    Bxw;��  �          @�zΐ33@�33�L(���33B�W
��33@��u�(�B��)                                    Bxw;��  �          @ȣ׿�  @�G��X���p�B����  @��R��
=�-p�B�33                                    Bxw;�F  
�          @ʏ\�{@�  ��  �733B�8R�{@�G�?���Ap�B�                                      Bxw;��  T          @�ff����@��H?���Az�B�33����@�{@W
=A��Bڊ=                                    Bxw;�  �          @��
���@�{?˅Ag�B��)���@��
@p��B�
B�=q                                    Bxw;�8  �          @�����@�G��#�
���B��
��@�
=�O\)�  B癚                                    Bxw<
�  �          @�=q�4z�@�  �!��ڏ\B�p��4z�@�p��L�����B�\)                                    Bxw<�  
�          @�{�3�
@�(���\)��G�B�\�3�
@���>�\)@5B�R                                    Bxw<(*  
Z          @��H�C�
@�����Q�B���C�
@�zᾮ{�]p�B�Ǯ                                    Bxw<6�  T          @���*�H@p  �N�R�\)B��*�H@�ff��ff��Q�B�B�                                    Bxw<Ev  �          @�ff�3�
@s�
�8����B�Ǯ�3�
@��
��(��K�B�                                     Bxw<T  �          @�\)�@��@����H�ə�B��R�@��@��׿&ff�љ�B�                                     Bxw<b�  �          @�(��G
=@��׿E����B�\�G
=@�\)?�G�A'�
B���                                    Bxw<qh  
�          @����_\)@�z�>�=q@'
=B����_\)@�p�@
=A�z�C E                                    Bxw<�  �          @�  �l��@�>��?�Q�C W
�l��@���?�A���C�f                                    Bxw<��  
(          @�\)�e@�Q�>�Q�@Z�HB�#��e@���@Q�A�G�C�                                    Bxw<�Z  �          @�\)�l(�@�{>�\)@,��C +��l(�@��@�A�G�C
=                                    Bxw<�   �          @�G��u@��>��H@�ffC�\�u@��\@(�A��C@                                     Bxw<��  �          @�p���{@�Q�@A��\CB���{@;�@\��B��C\                                    Bxw<�L  
�          @��H�~{@s�
@"�\A��C#��~{@%�@q�B�\C��                                    Bxw<��  �          @�33�l(�@�z�@G�A���C��l(�@S�
@aG�BQ�C
)                                    Bxw<�  "          @ƸR�e�@���@
�HA�=qC +��e�@Z=q@n{BCc�                                    Bxw<�>  �          @�33���@r�\@
�HA�z�C	�����@,��@[�B
�C�                                    Bxw=�  T          @�
=���@5@ffA�ffC�)���?�G�@O\)BG�CO\                                    Bxw=�  
�          @��R��z�@   @�\A���C�)��z�?�(�@C33A�33C#:�                                    Bxw=!0  
�          @�\)����@p�@�
A���CG�����?�Q�@>{A�
=C&��                                    Bxw=/�  �          @�z���=q@6ff?���A%�C����=q@\)@�A��\C+�                                    Bxw=>|  �          @�(���ff@O\)?��
Ag�C����ff@(�@'�A��C                                    Bxw=M"  "          @ƸR��ff@^�R@   A��HCz���ff@\)@H��A���C�                                    Bxw=[�  �          @��
����@b�\@$z�A�G�CO\����@@l��B
=CE                                    Bxw=jn  "          @�33����@vff@�A��
C	�f����@5�@S�
B�\Ck�                                    Bxw=y  T          @��H�~�R@�Q�?�
=A�=qC{�~�R@Vff@HQ�A�{C�q                                    Bxw=��  T          @�����33@^�R?�A�
=C�{��33@%�@=p�A�\C�R                                    Bxw=�`  
�          @\��33@u?��A�\)C
����33@:=q@E�A�\)C5�                                    Bxw=�  "          @�=q����@w�?�\A�{C
�����@=p�@Dz�A�p�C�                                    Bxw=��  �          @���xQ�@|��?:�H@���Cs3�xQ�@Z=q@Q�A�
=C
��                                    Bxw=�R  T          @��\�\)@^{�&ff�
=qB�G��\)@]p�?(��A�B�Q�                                    Bxw=��  �          @�=q�@��@Y��=u?G�C}q�@��@I��?�ffA���C��                                    Bxw=ߞ  �          @�녿�\)@��R��z��W33Bី��\)@�=q?
=q@ȣ�B��{                                    Bxw=�D  �          @���@�������hz�B܏\��@��?�@�  B�ff                                    Bxw=��  �          @�\)����@�ff��33���B�=q����@�
=>�  @*�HBՏ\                                    Bxw>�  �          @���.{@s�
�����X��B����.{@}p�>���@y��B��H                                    Bxw>6  "          @�\)�*=q@���xQ��B���*=q@�?^�RA��B�q                                    Bxw>(�  O          @��׿fff@��
�#33�ȸRB�
=�fff@���G���\)B�B�                                    Bxw>7�  �          @�ff�fff@��� ����=qB�B��fff@�33��G����RB�z�                                    Bxw>F(  
�          @��R�Q�@��R�\)����B�#׿Q�@��;����B��)                                    Bxw>T�  �          @���5@�G��!G���  B�W
�5@��H����p�B��                                    Bxw>ct  �          @����Tz�@�
=�Q���{B�8R�Tz�@��R���� ��B���                                    Bxw>r  �          @��Ϳ��@�
=�0  ��Q�B�� ���@��H�(����RB�Q�                                    Bxw>��  �          @Å�   @�Q��\(��	Q�B�\)�   @����
=�X��B���                                    Bxw>�f  �          @��H>B�\@��\�Vff��B�Ǯ>B�\@�{�����H��B�aH                                    Bxw>�  
�          @Å>B�\@�z��S33��RB��H>B�\@�
=��G��>=qB�u�                                    Bxw>��  �          @�ff>8Q�@�\)�Tz����B���>8Q�@�녿�  �:�RB��                                    Bxw>�X  T          @��>�33@�=q�^{�	=qB�  >�33@�
=�����Z{B�33                                    Bxw>��  
�          @�=q>��R@�z��dz��G�B��R>��R@��H��{�v�HB��f                                    Bxw>ؤ  "          @��>�z�@�z��n�R�{B�B�>�z�@�z��\��G�B�ff                                    Bxw>�J  T          @�ff?z�@�33�r�\�=qB�W
?z�@��
������B��3                                    Bxw>��  T          @�z�?�=q@��H�z�H���B�u�?�=q@�p��33����B�ff                                    Bxw?�  �          @���?z�H@�G��o\)��B�?z�H@�G��������B��R                                    Bxw?<  "          @���?�ff@�{�u��G�B��q?�ff@�\)�������B�8R                                    Bxw?!�  �          @�(�?�Q�@�{�qG����B�k�?�Q�@��R������
B�aH                                    Bxw?0�  �          @�=q?��@�p��n{�\)B���?��@�p���{��
=B�33                                    Bxw??.  �          @��
?��@���mp����B��R?��@�\)��=q����B�                                      Bxw?M�  �          @��?ٙ�@���vff�B�u�?ٙ�@��ff���
B�B�                                    Bxw?\z  
�          @�Q�?�p�@��p���z�B�  ?�p�@��R�   ��p�B���                                    Bxw?k   "          @���?���@���u��"=qB��?���@��
�ff��G�B��
                                    Bxw?y�  
(          @�Q�?�(�@s�
��ff�1z�Bzz�?�(�@����'���\)B��                                    Bxw?�l  
�          @�33�&ff@b�\�9�����BȨ��&ff@��\��G���{B���                                    Bxw?�  "          @�(���ff@6ff=�G�?�Q�C����ff@(��?�=qADz�C�{                                    Bxw?��  
�          @��R��  ?��?�p�AUC%G���  ?O\)?���A��\C*�                                    Bxw?�^  �          @����|(�����@L��B  CG\)�|(����@"�\A�ffCSW
                                    Bxw?�  �          @�=q��Q�Y��@@��B
��C?T{��Q��=q@"�\A�\)CKB�                                    Bxw?Ѫ  �          @�Q��w���\)@XQ�B�HCD+��w��p�@2�\B G�CQ��                                    Bxw?�P  
�          @���9�����@��BVG�CM�)�9���333@hQ�B'��C_��                                    Bxw?��  
�          @���N{��(�@�33BG�
CH�
�N{�#33@\(�B�
CZY�                                    Bxw?��  �          @���\)��@x��B0G�C;^��\)��@_\)B�CL\)                                    Bxw@B  �          @�����H>��
@n{B(��C/�{���H�u@fffB"ffCA8R                                    Bxw@�  �          @�  ��=q?��H@\��Bp�C#h���=q��@j=qB'��C4\)                                    Bxw@)�  �          @�
=��=q?��R@&ffA噚C:���=q?�  @G�B�RC'                                      Bxw@84  �          @��H��z�@3�
@ffAɮC�=��z�?���@J=qB�Cٚ                                    Bxw@F�  "          @��
���?�\@	��A�=qC.O\����u@��A�ffC6�f                                    Bxw@U�  �          @����(�?#�
@+�A�Q�C,�=��(�����@/\)A��C7xR                                    Bxw@d&  �          @�=q����?8Q�@��A�Q�C+�����׽��
@�A�z�C4޸                                    Bxw@r�  �          @����  ?�z�@z�A���C&�q��  >���@A�C/u�                                    Bxw@�r  
�          @�33��{?�\?�A���C E��{?��
@��A�C(:�                                    Bxw@�  T          @����@   ?У�A�z�C�H��?��@p�A��HC$�q                                    Bxw@��  "          @�  ���R?�@&ffA��C޸���R?Y��@C�
B�
C)B�                                    Bxw@�d  
�          @�G���G�?^�R?�(�A���C*:���G�>#�
@	��A��\C2�                                    Bxw@�
  �          @����u@_\)?���A�
=C	�q�u@+�@9��A�=qC�                                    Bxw@ʰ  �          @��H��G�@+�@G�A��HC�R��G�?��@A�B=qCh�                                    Bxw@�V  
�          @�z���{@��@��A�G�CO\��{?�z�@4z�AC#T{                                    Bxw@��  
�          @�����G�@\)@�A�z�C�=��G�?�Q�@.�RA�=qC �H                                    Bxw@��  �          @�z���@\)@�A�p�Cu���?�@.�RA�C#&f                                    BxwAH  �          @�33���
@.{?�{A��C�����
@�\@=qA�  C.                                    BxwA�  	�          @�=q���
?�33?�p�A��RC#�3���
?.{@�A���C,
                                    BxwA"�  
�          @�p����>�
=@Dz�B33C.�H������@B�\B��C;8R                                    BxwA1:  �          @�����
=?s33@L��Bp�C(����
=���@U�B�
C5�\                                    BxwA?�  �          @����ff?�ff@J=qB  C'p���ff�#�
@U�B�C4s3                                    BxwAN�  �          @�\)��
=?�(�@(�A�C#���
=?��@2�\A���C-#�                                    BxwA],  �          @�����  ?Ǯ@\)A�
=C"�f��  ?+�@7
=A�=qC,Y�                                    BxwAk�  T          @�����p�?�
=@-p�A�z�C#���p�>�@AG�A�p�C.k�                                    BxwAzx  T          @����?Ǯ@%�AӮC"s3��?&ff@<(�A��RC,��                                    BxwA�  �          @��H��?���@P  B�HC&k���<�@\(�B�C3��                                    BxwA��  
�          @�����?�z�@HQ�B33C&u����=�Q�@UB
�HC2�                                    BxwA�j  �          @������?�z�@Tz�B	G�C�����?�\@k�B�C-��                                    BxwA�  "          @��
���?�@(�A�G�C 33���?��@*=qA�{C(u�                                    BxwAö  T          @�33����?�33@/\)A��CW
����?k�@L��B�C)
                                    BxwA�\  �          @�{��{?�p�?�33A�
=C���{?��\@�A��C%��                                    BxwA�  �          @��\���@�?�A�\)C�)���?��R@=qẠ�C"Ǯ                                    BxwA�  "          @�G���
=@G�@�
A��HC޸��
=?�G�@%A�C%
                                    BxwA�N  T          @����?�p�?�33A�ffC �R��?�{@
=A��C'L�                                    BxwB�  "          @��H��p�?Ǯ?�33Ah��C#E��p�?��
?���A�z�C(��                                    BxwB�  
�          @������?�\)?�z�A��C%  ���?E�@ ��A�{C+c�                                    BxwB*@  "          @�33����?�Q�?�33A��C$Y�����?Y��@G�A���C*�f                                    BxwB8�  
�          @��R����?�{?�Q�Ai��C#
����?���?�\)A��C(��                                    BxwBG�  
Z          @����G�?��
?У�A��C#�)��G�?p��@�A�p�C)��                                    BxwBV2  
�          @�ff���R?�?�p�Ar�RC":����R?�\)?�
=A�p�C'�)                                    BxwBd�  "          @��
��G�@J�H?h��AC{��G�@.{?�{A��RC                                    BxwBs~  �          @�p�����@J=q?�A?\)C@ ����@(Q�@ffA�(�C�3                                    BxwB�$  
(          @�Q���@(Q�?�Q�AG\)C����@Q�?���A��C��                                    BxwB��  T          @�(���\)@!�<��
>8Q�C)��\)@��?L��A(�CQ�                                    BxwB�p  �          @�����=q@�þ�33�j�HC!H��=q@��>���@Z=qC�                                    BxwB�  �          @�����ff?�z�=���?}p�C����ff?��?.{@�\C!�                                    BxwB��  �          @�{��z�?�G�=�G�?��C!{��z�?��?&ff@��HC"G�                                    BxwB�b  �          @�
=��\)?�=q���
�uC#+���\)?�G�>��@�z�C#޸                                    BxwB�  �          @�33��(�?��H��G����C$���(�?��
<�>��
C#ff                                    BxwB�  �          @�(����H?��Ϳ#�
��G�C"�����H?�(���G�����C!Y�                                    BxwB�T  
�          @��\��=q?�\)�����P  C'}q��=q?�Q�L���	p�C$(�                                    BxwC�  �          @�  ���>Ǯ��33�z�RC/}q���?L�Ϳ�(��Xz�C*�
                                    BxwC�  �          @�G�����?��ÿ��H�S�
C'������?�녿Q���C$��                                    BxwC#F  �          @�=q���\?�G��p���"ffC&����\?��R��\��\)C#��                                    BxwC1�  
�          @������
?xQ�p���$(�C)G����
?��H�����G�C&�3                                    BxwC@�  
�          @��H��(�?z�H�����9�C)0���(�?�G��:�H��=qC&5�                                    BxwCO8  T          @�  �o\)@E?�33AUG�Cs3�o\)@%@�A�ffC=q                                    BxwC]�  "          @�{�S33@Vff?�p�A�(�C�{�S33@/\)@�A�  CE                                    BxwCl�  �          @��R�Vff@L(�?��
A��CaH�Vff@ ��@*=qB��C.                                    BxwC{*  
�          @�=q��@Dz�?�Q�AO\)C�R��@#�
@�
A�z�Cs3                                    BxwC��  �          @�
=��
=?�  ���R�\��C�f��
=@�\�&ff��C�)                                    BxwC�v  T          @�\)���?�  �����n�HC�3���@��B�\���CW
                                    BxwC�  
�          @�(���  ?����\��{C#����  ?�{�Ǯ����C�
                                    BxwC��  
�          @�Q���z�?z�H�p����HC'aH��z�?��Ϳ���p�C �                                    BxwC�h  "          @�ff��(�?\�����(�C���(�@
=�����{C��                                    BxwC�  
�          @�p��|��?�\)����  C �{�|��?�
=��=q��=qC�                                    BxwC�  �          @��\�\(�?��׿�Q���C!ٚ�\(�?�33�\���CQ�                                    BxwC�Z            @xQ��S�
?�(��������C���S�
?�\)��{��  C�                                    BxwC�   �          @z�H�\(�?��
��ff���C#J=�\(�?�����
=��=qC0�                                    BxwD�  �          @|���^�R?
=��G��ә�C*W
�^�R?�=q��  ��G�C"��                                    BxwDL  �          @�  �`  ?!G����
�ԸRC)Ǯ�`  ?��׿�G����C"#�                                    BxwD*�  �          @��e>�
=��
��C-O\�e?�G������ң�C$O\                                    BxwD9�  T          @��\�|��?\(������
=C'� �|��?���������\C޸                                    BxwDH>  �          @�ff�\)?^�R����  C'�\)?�G���p����HC@                                     BxwDV�  "          @����33?h�������{C's3��33?˅�z���=qC��                                    BxwDe�  �          @�p��|(�>�ff�p����HC-� �|(�?�33��R��C#�                                     BxwDt0  T          @�Q��|��?
=q�
=���C,8R�|��?��׿�{����C$                                    BxwD��  
�          @��R�s33>����\���C,�H�s33?�\)��
�ۙ�C#��                                    BxwD�|  
�          @�z��n�R>�
=��\����C-���n�R?���z���33C$\                                    BxwD�"  "          @����w
=>\��\��z�C.c��w
=?��
����=qC%)                                    BxwD��  
�          @�Q��`  >L����H�
��C0�=�`  ?c�
���� z�C%�q                                    BxwD�n  T          @��
�i��>#�
������C1���i��?W
=�\)��
=C'�                                    BxwD�  �          @����o\)>#�
�!G���HC1��o\)?aG�����G�C&�q                                    BxwDں  �          @��\��녾#�
�0  ��Q�C5�����?#�
�*�H��G�C,�                                    BxwD�`  �          @�33���
<��&ff���C3�����
?E���R��C)�3                                    BxwD�  "          @�{��Q쾙������G�C7��Q�>Ǯ�ff���
C/                                      BxwE�  
�          @�����
=?n{�S�
��C(0���
=?��:=q��Q�C��                                    BxwER  �          @�
=��(�?(��N{�Q�C,���(�?�G��:=q� ��C!                                      BxwE#�  �          @��H��(�?(��Dz��=qC,�\��(�?����1G����C"�{                                    BxwE2�  �          @��R���?^�R�C�
��\C)�����?��H�+���ffC 5�                                    BxwEAD  �          @�����?@  �Vff�C*����?��@  ��  C 
                                    BxwEO�  �          @�33��z�?�  �Tz��
�\C$�H��z�@���5����
C8R                                    BxwE^�  "          @�z���  ?���:�H��(�C(
=��  ?�� ���ɅCǮ                                    BxwEm6  "          @�z����?�{�1���\)C(33���?����
=��{C ��                                    BxwE{�  "          @�������G���R����C5)����?���=q���HC-�H                                    BxwE��  �          @����
=@333�J=q��C�f��
=@e��  ��CO\                                    BxwE�(  "          @�����@���Dz���\)Cc�����@I�������C�                                    BxwE��  �          @�p����\@.{�-p�����C�)���\@W�������33C{                                    BxwE�t  "          @���=q@&ff������RC޸��=q@J�H��=q�l��C�                                    BxwE�  �          @�Q����@{�'
=�ŅC8R���@Fff������\)CǮ                                    BxwE��  
�          @�ff���
@���)����
=C33���
@:=q������Ch�                                    BxwE�f  �          @�{����@��G
=��{C�H����@H������p�C��                                    BxwE�  �          @Å��{?�
=�l����C%޸��{@
=q�N�R� {CE                                    BxwE��  "          @�ff���?�p���
=�,  C$L����@ff�n{��C�R                                    BxwFX  
�          @Ǯ��z�?����ff�)ffC"n��z�@   �i���z�Cff                                    BxwF�  �          @�\)��33?�{�x����C �R��33@'��S33���CW
                                    BxwF+�  "          @�����?�z����R�(�
C"ff��@ ���j�H�
=Cu�                                    BxwF:J  �          @�  ��p�?�(��w
=�{C"�\��p�@{�S�
� Q�C�                                    BxwFH�  �          @�\)��ff?�\)�QG����CQ���ff@,(��(Q��ȏ\Cs3                                    BxwFW�  "          @�p���  @G��>{����C�R��  @0����
���C)                                    BxwFf<  "          @����\@��5����
CǮ���\@1G��	����33Cn                                    BxwFt�  T          @�  ��@�R�)����
=C����@7
=��Q���
=C\                                    BxwF��  
�          @�����33@��!G����CǮ��33@+���{��(�C\)                                    BxwF�.  �          @��H����?&ff� �����RC-�\����?���\)����C'=q                                    BxwF��  �          @������\?�p��.{��  C'�����\?���\����C ��                                    BxwF�z  �          @��
��(�?�Q�������C!
=��(�@{��33�p  CL�                                    BxwF�   �          @�������@1��333��{C�����@[�������  C�f                                    BxwF��  �          @������?�\)�����33C$O\����@
=q��Q��tQ�C�=                                    BxwF�l  �          @�z���p�?�����(�C"� ��p�@ff���
��33CxR                                    BxwF�  �          @�z�����@��������HC������@8Q쿵�M�Cs3                                    BxwF��  T          @�z���@�\)�G�����C�3��@���ff�  C��                                    BxwG^  �          @�
=��G�@�����R��33C�H��G�@����  ��C�                                    BxwG  �          @�(����@��\��R��(�C�H���@��ÿ����C�f                                    BxwG$�  �          @�\)���R@Z�H� ������C}q���R@~{��G��c33C
T{                                    BxwG3P  �          @Å����@g��p����C�����@�=q���2�HC��                                    BxwGA�  �          @Å��(�@}p����=qC0���(�@��
�u�G�C\)                                    BxwGP�  T          @��H��{@s33�\)����C	Ǯ��{@�Q쿓33�.�HC��                                    BxwG_B  �          @�Q���  @mp���
���\C
����  @��
��G���RC��                                    BxwGm�  T          @�\)��(�@n�R�
�H��z�C	�f��(�@���{�+\)C��                                    BxwG|�  �          @��
�\)@u���Q���Q�C:��\)@�ff�\(��
=C��                                    BxwG�4  �          @��
�e�@������'�
C �{�e�@��>��?�G�B��                                    BxwG��  �          @�p����H@hQ��  ��33C{���H@}p��:�H��33C	�)                                    BxwG��  �          @��k�@�
=�}p��!G�C
�k�@��\>\)?��HCW
                                    BxwG�&  �          @�G��e�@�{��G���G�C��e�@��?+�@߮C��                                    BxwG��  �          @���S33@�\)�(����G�B��)�S33@�Q�>�@�B��                                    BxwG�r  "          @�
=�QG�@�Q쿃�
�.=qB�
=�QG�@�(�=�G�?�B��                                     BxwG�  T          @�  �O\)@��ÿ�\)�;�
B�33�O\)@�p�=#�
>�p�B�ff                                    BxwG�  �          @����a�@��8Q���  C0��a�@�
=>Ǯ@���C�f                                    BxwH d  �          @���]p�@�ff������B��q�]p�@�{?#�
@�
=B��f                                    BxwH
  �          @���N{@�녿Y�����B���N{@��
>�(�@�{B��                                    BxwH�  �          @��\�0  @�  �����*{B�=�0  @��>��@!�B�                                    BxwH,V  
�          @��\�=p�@�33����O�B���=p�@��׼��
�.{B�#�                                    BxwH:�  �          @��{@��Ϳh���  B�3�{@��R>�G�@��RB�33                                    BxwHI�  �          @�ff�8��@����z�H��B�  �8��@���>�\)@5B�
=                                    BxwHXH  
�          @���g
=@���\)���\CG��g
=@��?
=@�{CO\                                    BxwHf�  
�          @�����ff@�������ap�BՀ ��ff@��\<#�
>�B�u�                                    BxwHu�  T          @����(�@�
=��G��%�B�Q��(�@��>�z�@@  B�                                     BxwH�:  
�          @�p��0��@�(��.{��(�B���0��@�z�?
=@�  B���                                    BxwH��  �          @���'�@��R�!G����
B�Ǯ�'�@��R?(��@�B���                                    BxwH��  T          @�  �5@�ff����ffB�\)�5@�=q?���A4(�B�3                                    BxwH�,  "          @�\)�s�
@w
=?!G�@�33C�H�s�
@dz�?˅A��C�{                                    BxwH��  T          @�����@dz�?��@��HC
h�����@S33?�p�A~ffC�)                                    BxwH�x  �          @�=q�|(�@z�H>��@(��C(��|(�@n�R?��RAM�C��                                    BxwH�  T          @����G�@��>k�@C�H��G�@w�?�  AI��CB�                                    BxwH��  �          @�\)��  @�33>8Q�?���CG���  @z�H?��HAB�HC�\                                    BxwH�j  
�          @����\)@�z�>�@�=qC���\)@x��?��RAn�HC�                                    BxwI  T          @�ff��\)@`��>.{?�p�C�H��\)@W
=?�ffA)C!H                                    BxwI�  
Z          @�{����@X��>���@�Q�C\)����@K�?�p�AHz�C{                                    BxwI%\  �          @�(����R@j=q?B�\@�  C����R@Vff?�33A�Q�C��                                    BxwI4  
�          @�p���@n{?Y��A	C
^���@X��?�  A�  C�q                                    BxwIB�  �          @�p���\)@Vff?n{A�RC33��\)@@��?�p�A�z�C�                                    BxwIQN  T          @�ff����@Q�?�=qA/�C\����@:=q?�\)A�
=CG�                                    BxwI_�  
�          @����33@N�R?�Q�Amp�Cn��33@1G�@(�A�=qC�                                    BxwIn�  �          @�\)�y��@]p�?�\)A�C
^��y��@<��@�HA���C�
                                    BxwI}@  �          @��Ϳ�G�@���#�
��B՞���G�@��H?�(�AYG�B֞�                                    BxwI��  �          @��\��{@�����Q�B�\��{@�G�?��AMBʞ�                                    BxwI��  �          @���=q@\����(�B�#׿�=q@�ff?�  A:{BȊ=                                    BxwI�2  �          @�
=��Q�@��׿L����{B�\)��Q�@���?5@��B�Q�                                    BxwI��  "          @ə����
@��R�fff�\)B�G����
@��?��@�ffB��                                    BxwI�~  �          @�33���@�=q<��
>W
=B������@�z�?���AUp�B��                                    BxwI�$  
�          @�z���@�녽��
�E�B��\��@��?��AEB�                                    BxwI��  T          @�p���{@�(��n{��B�  ��{@�?
=q@��HB���                                    BxwI�p  T          @����
=@���L�;��B�Q��
=@�  ?��AI�B�                                     BxwJ  
�          @�Q��6ff@���>��?���B����6ff@�=q?�
=A]p�B��                                    BxwJ�  �          @�ff�W�@�p�?333@�  B�#��W�@��H?�Q�A��
B���                                    BxwJb  	�          @�33�g
=@�(�?�G�A�RB�B��g
=@�\)@�RA�ffB��3                                    BxwJ-  T          @�z��Z=q@�G�?Y��@��B�.�Z=q@�@33A�{B�(�                                    BxwJ;�  �          @�(��\(�@�
=?\)@��RB�#��\(�@�ff?�Q�A��B�u�                                    BxwJJT  
�          @�  �b�\@�\)?��\A  B����b�\@��H@Q�A��\C.                                    BxwJX�  �          @�z��=p�@���?B�\@�B���=p�@�@ ��A�ffB���                                    BxwJg�  �          @�p��   @��?333@�p�B�Ǯ�   @��\?�A�p�B�R                                    BxwJvF  "          @������@�?�33A)C� ���@�?��HA|��C��                                    BxwJ��  �          @������
@�?���AU�C!H���
?���?�p�A��C ٚ                                    BxwJ��  "          @�
=���@{?�=qAjffCh����?�\@�
A���C"p�                                    BxwJ�8  �          @�(����?�\?��Ai��C"�����?���?�A�ffC&�)                                    BxwJ��  T          @Å���?��?���Ar�\C#޸���?��H?�Q�A�ffC'��                                    BxwJ��  "          @����\)@(�?��Ao\)C����\)@   @�A��C�                                    BxwJ�*  "          @�z���33?޸R?޸RA��
C!����33?��\@
=A��C&��                                    BxwJ��  �          @��R��p�?��H?�A�=qC!����p�?��\@�A���C&=q                                    BxwJ�v  �          @�33��=q@��?�
=A��C�{��=q?�p�@	��A��C!!H                                    BxwJ�  "          @�
=��{@*=q?��AI�C�
��{@�?�33A�z�CQ�                                    BxwK�  �          @������\@G
=?���A"�\C}q���\@1G�?��
A��\Cc�                                    BxwKh  T          @�(����H@O\)?��A��C}q���H@9��?��
A�G�CJ=                                    BxwK&  
Z          @Å���\@j�H?#�
@���CǮ���\@Z=q?��RAbffC�                                     BxwK4�  
�          @����=q@e?5@أ�CW
��=q@Tz�?��Al(�CxR                                    BxwKCZ  �          @�  ���@g
=>�Q�@^{C���@Z�H?���A9�C:�                                    BxwKR   
�          @����=q@��
���R�E�C����=q@��\?+�@�(�C�                                    BxwK`�  T          @�  ��Q�@S�
=L��?�C�R��Q�@L��?Tz�Az�C��                                    BxwKoL  �          @�Q�����@0��>#�
?У�CO\����@(��?O\)AG�CT{                                    BxwK}�  
�          @�
=��p�@�\?��@�G�C���p�@ff?���A-C�                                    BxwK��  �          @�����
=@�?333@��C  ��
=@ff?�p�AEG�C#�                                    BxwK�>  "          @������H?���@33A�
=C(O\���H?
=@  A�{C-��                                    BxwK��  
�          @�
=��Q�>�  @K�B�C1)��Q�   @I��B 33C9�                                    BxwK��  �          @����Q�?�G�@�RA��
C)�f��Q�>�@=qA�(�C/B�                                    BxwK�0  �          @�Q���  ?���?�(�A;\)C(0���  ?c�
?�(�Ab{C+@                                     BxwK��  T          @�z���p�?�?�\A�z�C%T{��p�?xQ�@33A�C)��                                    BxwK�|  
�          @��
��z�?�33?���A}�C"����z�?��R?�Q�A�z�C'
                                    BxwK�"  �          @�Q���p�?�(�?���AxQ�C ���p�?��@ ��A�\)C$�                                    BxwL�  �          @�Q���=q@��?���AJ�RCQ���=q@z�?���A��C��                                    BxwLn  �          @�p���{@'
=?�\)A.{C\)��{@�?�Q�A���CW
                                    BxwL  T          @�{��ff@+�?��A!�C��ff@
=?��A�C�\                                    BxwL-�  �          @�����
@1G�?z�HA��C�
���
@{?���Az�RCE                                    BxwL<`  �          @������R@.�R?��A.�HC.���R@��?�Q�A���C(�                                    BxwLK  T          @�  ��G�@�\?�(�An{C����G�?��?��HA�  Cp�                                    BxwLY�  �          @�����?�ff?�(�A�  C#�����?�{@�\A�Q�C(�                                    BxwLhR  "          @�  ��=q?Tz�@#33A�33C*� ��=q>aG�@*�HA�(�C1��                                    BxwLv�  
(          @��R�!�@s�
?�{A���B�{�!�@U@(�A�B�33                                    BxwL��  �          @��\�u@�{?(��@���B�8R�u@�z�?���A���B�G�                                    BxwL�D  
�          @�=q�n{@�\)>�=q@%�Bų3�n{@�Q�?��Az�HB�\)                                    BxwL��  �          @�녾\)@�G�>Ǯ@k�B����\)@�G�?�G�A�G�B�Ǯ                                    BxwL��  �          @�Q�z�@��H����#�B�p��z�@�>���@;�B�B�                                    BxwL�6  �          @�G���ff@\�   ���B�ff��ff@���?fffA(�B�u�                                    BxwL��  "          @��Ϳ!G�@��H��
=�n�RB�W
�!G�@ȣ�?��A=qB�z�                                    BxwL݂  
�          @�{��Q�@��=u?�\B�k���Q�@�\)?\AZ�\B���                                    BxwL�(  "          @Ϯ�#�
@�\)>.{?�p�B��q�#�
@ȣ�?��Aj=qB�                                    BxwL��  T          @�Q쾣�
@�
=?
=@�ffB��3���
@�p�@33A���B���                                    BxwM	t  T          @θR��@��������B�.��@���?��A;�
B�aH                                    BxwM  "          @�{�.{@��ͽ����B�33�.{@�Q�?��A@  B�z�                                    BxwM&�  �          @�(����@��?p��A�
B˸R���@���@�
A�Q�B�8R                                    BxwM5f  "          @�G��
=@�  �#�
���RB�p��
=@��H?�Q�AK�B��3                                    BxwMD  �          @ҏ\���@У׿#�
���
B��\���@Ϯ?Y��@���B���                                    BxwMR�  �          @љ�����@Ϯ�&ff��{B�#׾���@�
=?W
=@�33B�(�                                    BxwMaX  T          @�  ��R@�{���H����B���R@�z�?z�HA
=B��                                    BxwMo�  �          @��þ�\)@�  ��\���B�  ��\)@�ff?xQ�Az�B�
=                                    BxwM~�  T          @ҏ\��ff@�논��k�B��;�ff@�z�?���AL(�B�                                      BxwM�J  �          @�p��   @�z�B�\��{B����   @�Q�?��A5B�Ǯ                                    BxwM��  
�          @�  ���
@�{�.{��Q�B�p����
@�?Y��@�
=B�u�                                    BxwM��  T          @��H�@  @׮�h����(�B��@  @أ�?!G�@���B��3                                    BxwM�<  
�          @׮�z�@�ff�����4z�B�녿z�@�33?�
=A!p�B�\                                    BxwM��  T          @�\)�&ff@�{��p��G�B�
=�&ff@�33?��Az�B�.                                    BxwMֈ  �          @׮�:�H@�{�k���p�B��=�:�H@�=q?��\A.{B�                                    BxwM�.  T          @Ϯ�p�@��
?��HA�ffB����p�@xQ�@(��A�G�B��
                                    BxwM��  �          @�33��\)?�{@]p�B=qC�\��\)?��@r�\B�C(@                                     BxwNz  T          @�{��\)@   @e�B=qC)��\)?��@���B��C �H                                    BxwN   �          @������?�@vffB�C�����?��\@�BQ�C(Q�                                    BxwN�  �          @У����
@
=@{�B�C�����
?�@��B(33C"��                                    BxwN.l  �          @������H?�  @\)B{C (����H?O\)@���B$=qC*s3                                    BxwN=  �          @�G���ff?��\@��HB
=C(J=��ff=�Q�@�
=B!C2�R                                    BxwNK�  �          @�ff���R@p  @$z�A��C
T{���R@E@UB\)C�=                                    BxwNZ^  �          @�G����H@R�\@H��A�C�\���H@ ��@r�\Bp�C�                                    BxwNi  T          @ə����R@q�?�Q�A��\C�����R@P��@.�RA�=qC�)                                    BxwNw�  
�          @�G���@���?�(�AY��C�{��@n�R@ffA�
=C�                                    BxwN�P  T          @�=q��(�@n�R@'�A�{C�=��(�@C�
@XQ�Bz�C�                                    BxwN��  T          @�������@r�\@N{A�G�C�=����@>�R@~�RB�\C��                                    BxwN��  �          @�������@mp�@S�
A�33C	\)����@8��@���B  Cc�                                    BxwN�B  "          @�z��s�
@_\)@p  B  C	���s�
@$z�@�B/��C��                                    BxwN��  �          @θR�a�@hQ�@|��B�RC(��a�@*�H@���B9�C޸                                    BxwNώ  �          @�  �1�@��@8��Aԏ\B홚�1�@�G�@{�B(�B��
                                    BxwN�4  �          @���:�H@�\)@#33A�z�B�\)�:�H@�G�@h��B
=B��                                    BxwN��  �          @�G��C�
@�G�@33A���B�\�C�
@���@Z=qA��
B���                                    BxwN��  �          @Ϯ�:=q@���@8��A��
B���:=q@�Q�@{�B{B��\                                    BxwO
&  
�          @�p��1�@�\)@<(�A�  B�ff�1�@��R@}p�B�B��                                    BxwO�  T          @��
�"�\@�{@)��AŅB�33�"�\@�
=@n�RBB�(�                                    BxwO'r  "          @�Q��9��@�G�@�A�\)B�k��9��@�(�@a�B�\B��                                    BxwO6  "          @��
���@��\@	��A�Q�B�
=���@�
=@Tz�A�ffB��                                    BxwOD�  "          @�33�R�\@��R@ffA�G�B�.�R�\@�(�@H��A�=qB��f                                    BxwOSd  �          @θR�0��@���?�A��RB�{�0��@��@@��Aޏ\B��                                    BxwOb
  
�          @�33�8��@�p�?��A`��B�(��8��@�ff@,(�A��HB��\                                    BxwOp�  
�          @˅�(�@�?�Q�ARffB��(�@�
=@)��Ař�B�{                                    BxwOV  
�          @˅�(��@���?�=qAf�RB����(��@�=q@0��A�{B��                                    BxwO��  �          @�{�)��@��H?�\A~�HB��)��@�=q@<��A�{B�B�                                    BxwO��  "          @��H�Q�@�(�?�G�AxQ�B�#��Q�@�33@@  A�G�B�                                    BxwO�H  
�          @�=q��@��H?�\)Ad  Bծ��@��H@:=qA�ffB؅                                    BxwO��  �          @�  ���@��?�  AUp�B��H���@��H@2�\A�G�B׀                                     BxwOȔ  �          @�
=��@��H?�G�Az�B�
=��@�
=@z�A�(�B�#�                                    BxwO�:  �          @�G��
�H@���?�z�A#
=B�u��
�H@���@��A�(�B�                                      BxwO��  T          @�  ��(�@�?�(�Av�RBؽq��(�@�p�@>�RAڏ\B�                                      BxwO�  
�          @��H�ff@��@{A�B�L��ff@�{@XQ�B �RB���                                    BxwP,  �          @��Ϳ�  @��
?�{A!G�B�#׿�  @��@�HA�33B̨�                                    BxwP�  �          @��
�8Q�@�G�>�@���B��{�8Q�@�G�?��A�(�B��                                    BxwP x  T          @��H>�  @љ��333�\B���>�  @�G�?=p�@ϮB���                                    BxwP/  �          @�z�?O\)@��
��=q�\��B�z�?O\)@�녾����=qB��f                                    BxwP=�  �          @�z�?�33@�����R��G�B�\)?�33@˅�^�R���B��                                    BxwPLj  "          @�
=?�{@�����\��B���?�{@�z�n{��{B���                                    BxwP[            @�
=?���@��
�   ����B��
?���@����R���B�=q                                    BxwPi�  
�          @�\)?�\@�z�xQ��	B���?�\@�ff>Ǯ@]p�B�#�                                    BxwPx\  "          @У�?�33@�G���=q�_�
B�\?�33@Ǯ�k��   B��                                    BxwP�  �          @�\)?�z�@�p���=q��
=B�(�?�z�@�p��   ��p�B��\                                    BxwP��  �          @�?��H@�=q������\B��
?��H@��H�z����B�aH                                    BxwP�N  �          @�p�@p�@�G��ff��
=B�\@p�@���Y�����
B�p�                                    BxwP��  �          @�@{@����(���G�B�Ǯ@{@��
�p����\B�L�                                    BxwP��  �          @�@(�@����
��
=B���@(�@���ff��RB�p�                                    BxwP�@  "          @˅@�@�ff��ff�b�\B��\@�@��;�\)�"�\B��                                    BxwP��  �          @��H@-p�@�  �����j=qBG�@-p�@�
=��p��W
=B�\)                                    BxwP�  "          @�p�@W
=@��Ϳ�\)�k�BcG�@W
=@��
����Bg��                                    BxwP�2  "          @�@U�@��\��Q���
BQ�@U�@����N{���
Bdff                                    BxwQ
�  	�          @�
=@\)@e�������B'�@\)@�(��L(���z�B>��                                    BxwQ~  T          @љ�@�z�@s�
�Tz���{B*p�@�z�@�{�=q��p�B;�H                                    BxwQ($  
�          @��@�p�@���8�����B3�
@�p�@�{����=qBAz�                                    BxwQ6�  "          @�ff@\)@����z���{BB(�@\)@�p����
�733BK��                                    BxwQEp  T          @�{@n�R@���	����p�BQ  @n�R@����ff�ffBX                                    BxwQT  "          @�=q@���@����!G���Q�BB�@���@����p��O�BLz�                                    BxwQb�  "          @�33@�Q�@�G��@����33B;�H@�Q�@��H� ����  BI�\                                    BxwQqb  T          @ҏ\@���@�p��,����=qB1  @���@����p��t��B=�\                                    BxwQ�  T          @��@��\@����8���ͅB.��@��\@�p�����(�B<��                                    BxwQ��  �          @��@�\)@l���Z=q��RB(�@�\)@�33�!����
B*p�                                    BxwQ�T  �          @��
@��\@b�\�W�����B(�@��\@�{�!G���=qB#��                                    BxwQ��  "          @ָR@�ff@xQ��3�
��=qBG�@�ff@��Ϳ�z���{B,z�                                    BxwQ��  �          @���@�G�@o\)�@���ң�B�R@�G�@����Q���Q�B'�                                    BxwQ�F  �          @�Q�@��H@p���6ff�Ǚ�B\)@��H@��ÿ�(����B&(�                                    BxwQ��  �          @�(�@��@U��W����B�@��@~�R�$z���p�B�                                    BxwQ�  �          @���@�@Z=q�Z�H��B
�\@�@�=q�&ff���B                                      BxwQ�8  �          @�z�@��H@c�
�X����B\)@��H@��R�"�\����B$
=                                    BxwR�  4          @�ff@�Q�@U�]p�����B�R@�Q�@�Q��*=q���
B��                                    BxwR�  �          @��@��
@K��XQ���Q�A��R@��
@u��'
=��BQ�                                    BxwR!*  �          @��H@�p�@C33�S�
��33A�(�@�p�@l(��%����\B�                                    BxwR/�  �          @�\)@�ff@g��U����B"��@�ff@�Q���R��=qB5z�                                    BxwR>v  �          @�p�@��@L���Tz�����B��@��@u�#33���B�\                                    BxwRM  �          @�  @�=q@.{�I����z�A؏\@�=q@U��\)����B \)                                    BxwR[�  �          @���@�@B�\�H�����HA�@�@i���=q���\B                                    BxwRjh  �          @�=q@��\@C33�W���RA��
@��\@mp��(Q���\)Bp�                                    BxwRy  �          @�ff@�33@5��S33���A�Q�@�33@^�R�'
=��G�B	(�                                    BxwR��  T          @أ�@�Q�@@���W
=���A�  @�Q�@j=q�(Q�����B                                    BxwR�Z  T          @�\@��
@aG��>�R��
=B@��
@��\�	����G�B�R                                    BxwR�   T          @�=q@�=q@��R�(�����B$�@�=q@���z��^�\B0��                                    BxwR��  �          @��@���@�ff�ff��{BR��@���@��׿^�R��ffBY(�                                    BxwR�L  �          @�(�@�Q�@������RBM�@�Q�@�z�c�
��RBS                                    BxwR��  �          @�
=@�(�@�G����p��BP
=@�(�@�녿�R��33BUp�                                    BxwRߘ  T          @�\)@�
=@����G���z�BG  @�
=@��H�Tz���z�BMz�                                    BxwR�>  �          @�p�@���@������}G�BC�\@���@�\)�G�����BI�                                    BxwR��  �          @�=q@���@�����
��p�B?�@���@�33�k���  BF�
                                    BxwS�  �          @�R@���@�p�������BCQ�@���@��׿�G�� ��BJ�                                    BxwS0  �          @��@�  @�ff�z����HBK  @�  @��׿c�
��G�BQ                                    BxwS(�  �          @�\)@���@��
�
=����BB{@���@�Q쿙�����BJp�                                    BxwS7|  �          @�p�@�  @��H�#�
���B;��@�  @��׿�33�-�BE�                                    BxwSF"  �          @�G�@�p�@�G��*�H���BB\)@�p�@�  �����1BK��                                    BxwST�  �          @�@��@�z��/\)��
=BMQ�@��@����  �:�RBVz�                                    BxwScn  �          @陚@�{@�33�)����Q�BO�
@�{@������4(�BX�                                    BxwSr  �          @��@��@���#33����BN�@��@�������.ffBV�
                                    BxwS��  �          @�\@�
=@�������1BF{@�
=@���8Q��  BI��                                    BxwS�`  �          @�  @�G�@�Q��ff��Q�BQ��@�G�@��H�fff��
=BX��                                    BxwS�  �          @޸R@���@�p��'���
=BJ��@���@��
���R�G�
BT\)                                    BxwS��  �          @�=q@|(�@�z��#�
��p�B\p�@|(�@�=q���
�   Bd{                                    BxwS�R  �          @���@xQ�@�{����
B^��@xQ�@�녿���  BeQ�                                    BxwS��  �          @�G�@�p�@�G���\��{BT
=@�p�@��Ϳ��
�p�B[                                      BxwS؞  �          @陚@�  @�{�����BO��@�  @��\������\BW�                                    BxwS�D  �          @�(�@�{@���z�����BJz�@�{@�G�����  BR                                      BxwS��  �          @��H@�z�@�{�{���RBL\)@�z�@�G��z�H���BS\)                                    BxwT�  �          @�z�@�z�@�Q��������BM�
@�z�@���s33��(�BT��                                    BxwT6  �          @�G�@���@�������=qBJQ�@���@��������\BQ��                                    BxwT!�  �          @���@���@��
��\�w�BF�
@���@��B�\���BL�H                                    BxwT0�  T          @�Q�@�33@������HBI�@�33@�(��Q��ȣ�BO��                                    BxwT?(  �          @��H@�33@�
=��
����BB�\@�33@�33��{��BJz�                                    BxwTM�  �          @��@�@�{�����z{B?@�@���E����
BF33                                    BxwT\t  �          @�(�@�
=@�
=?��A&�RB
=@�
=@�=q@�A��B                                    BxwTk  �          @�@��R@~{?��HA|z�B33@��R@\(�@2�\A�\)A��R                                    BxwTy�  �          @�(�@�p�@e@33A��B\)@�p�@@  @A�A�  A��                                    BxwT�f  �          @��@���@���?�\)A,Q�B�@���@w
=@G�A���B33                                    BxwT�  �          @�{@���@�z�?�\)APz�B�H@���@k�@\)A�  B\)                                    BxwT��  �          @�\@���@�Q�?�  A<��B��@���@u�@��A��RBp�                                    BxwT�X  �          @��H@���@��
?�AffBG�@���@�Q�@ffA��HB
�H                                    BxwT��  �          @�Q�@��
@��\?�Q�A8  B�\@��
@z=q@
=A�(�B\)                                    BxwTѤ  �          @�{@�{@�p�?˅AL��BQ�@�{@}p�@!G�A�ffB=q                                    BxwT�J  �          @�33@�@���?���Av=qBQ�@�@\)@8��A��B=q                                    BxwT��  �          @�G�@��R@��?��
AaB�R@��R@\)@.{A�\)B��                                    BxwT��  �          @�\)@���@�z�?�A5��B=q@���@~{@ffA�(�B(�                                    BxwU<  �          @��
@��@���?�Ar{B�H@��@\)@7�A��B�                                    BxwU�  �          @�(�@��@���?�
=Ar�\Bz�@��@w
=@6ffA��
B
Q�                                    BxwU)�  �          @�G�@��R@���@   A~�RBff@��R@vff@:�HA��HB�R                                    BxwU8.  �          @��@�z�@��@	��A��B(��@�z�@�=q@G�A�B\)                                    BxwUF�  �          @�{@��@�z��G�����BM\)@��@��R�W
=��\)BT
=                                    BxwUUz  �          @�Q�@r�\@�33��Q��
=BU�\@r�\@�33�1����Bdp�                                    BxwUd   �          @�ff@z�H@�=q���
�Q�BK�@z�H@���<�����RB\G�                                    BxwUr�             @��@��@��������p�BC�
@��@����8����
=BT                                    BxwU�l  �          @�G�@��@�p������ ��BI��@��@��5���=qBY�H                                    BxwU�  �          @���@z�H@��������=qBP33@z�H@�G��3�
��\)B_�                                    BxwU��  �          @��@�Q�@�Q���Q��G�BG\)@�Q�@����7
=���BX{                                    BxwU�^  �          @�=q@x��@�(������BG��@x��@��>{��  BY�                                    BxwU�  �          @�(�@�(�@�Q����
��B>�@�(�@�=q�@������BP�\                                    BxwUʪ  �          @��@���@�����  �Q�B:�@���@��
�L(��Џ\BN��                                    BxwU�P  �          @�p�@�  @�ff�|(��p�B@33@�  @�
=�6ff���BR{                                    BxwU��  �          @�@z�H@�Q��vff�(�BC�@z�H@�Q��0  ���BU
=                                    BxwU��  �          @�(�@|(�@����u���HBDff@|(�@�G��.{���BU=q                                    BxwVB  �          @�
=@��@����w
=���B:33@��@����1���  BL{                                    BxwV�  �          @�z�@���@��xQ����B0�@���@�{�5���
BD
=                                    BxwV"�  �          @�\)@|(�@k������&
=B,G�@|(�@���u�� ffBG�                                    BxwV14  �          @�z�@~�R@����\)�	
=B=  @~�R@��\�:�H����BO�                                    BxwV?�  �          @�
=>�p�@�z�fff���B���>�p�@�p�?(�@�ffB��                                    BxwVN�  �          @�p�?Q�@��ÿp����RB�?Q�@ҏ\?��@��RB��)                                    BxwV]&  �          @�z�?���@θR������
B���?���@�G�>��@dz�B���                                    BxwVk�  �          @��?�@�z΅{�<��B�=q?�@�G�=�G�?}p�B��                                    BxwVzr  �          @�ff?��H@�  �����=qB�B�?��H@�z�}p���B�{                                    BxwV�  �          @�{@p�@�33���B�k�@p�@��8Q�����B��\                                    BxwV��  �          @ۅ@'
=@�ff�
�H���B��@'
=@�G��E���ffB���                                    BxwV�d  �          @ٙ�@G
=@���(�����Bs�@G
=@��R�\(���\By�R                                    BxwV�
  �          @��@`��@�\)��H���B[(�@`��@��Ϳ�p��,��Bc�R                                    BxwVð  �          @�(�@_\)@�\)�*=q���HB\
=@_\)@�ff���H�K\)Be��                                    BxwV�V  �          @�(�@^{@�(��7
=���BZp�@^{@��Ϳ��i�Be(�                                    BxwV��  
�          @љ�@)��@�p��>�R��
=B{z�@)��@��R��p��uG�B�8R                                    BxwV�  �          @�(�@0  @�  �W���\Bt�R@0  @�(��	����33B�                                    BxwV�H  �          @�=q@/\)@�
=�L����G�Bt��@/\)@�=q���R���B=q                                    BxwW�  �          @�@(Q�@��R�Y��� �
Bs33@(Q�@���\)��Q�Bff                                    BxwW�  �          @θR@C�
@�=q�xQ��=qBT\)@C�
@��H�6ff�ӅBg
=                                    BxwW*:  �          @�@B�\@j�H��=q�(BIQ�@B�\@���W�����Ba�                                    BxwW8�  �          @ə�@@  @g���ff�'  BIz�@@  @���P����
=B`�                                    BxwWG�  �          @��
@B�\@a����H�,  BD��@B�\@��Z�H���B^{                                    BxwWV,  �          @��@=p�@l������)Q�BMff@=p�@��H�Vff��ffBd��                                    BxwWd�  �          @��H@%�@xQ������*(�Ba�\@%�@�Q��QG���G�Bv(�                                    BxwWsx  �          @�Q�@@  @��R�O\)���Be�@@  @��\����  Br\)                                    BxwW�  T          @�G�@<(�@��R�33��z�Bv�H@<(�@�G��=p��У�B|Q�                                    BxwW��  T          @ʏ\@�\@��
�#�
��B�k�@�\@��R?��AD  B�\)                                    BxwW�j  T          @�33@
�H@�{>�{@G
=B��@
�H@��R?ٙ�AyB�aH                                    BxwW�  T          @ʏ\?�Q�@�  >�@�(�B�8R?�Q�@��?�=qA�ffB��                                    BxwW��  �          @˅@ff@�
=?��@��B�=q@ff@�?�A��\B�p�                                    BxwW�\  �          @��H?��@���?��@��RB�aH?��@�\)?�
=A��B��q                                    BxwW�  �          @��?�@\    <��
B�G�?�@�p�?�z�AN�RB�u�                                    BxwW�  �          @ʏ\?�@�녾������B��q?�@�?�G�A8Q�B�
=                                    BxwW�N  �          @���?�
=@\�\)��=qB�� ?�
=@�{?��\A;�
B��                                    BxwX�  �          @�  ?z�H@�p���G���G�B��?z�H@���?�=qADQ�B��                                    BxwX�  �          @Ǯ?��@��
���R�5�B��)?��@���?�\)A&ffB��=                                    BxwX#@  �          @�  @z�@��
�   ���HB�#�@z�@��\?aG�A{B��)                                    BxwX1�  �          @�\)@33@�33�!G���Q�B�p�@33@��\?B�\@߮B�W
                                    BxwX@�  �          @Ǯ@p�@�{�\)��(�B�Q�@p�@�p�?J=q@�G�B��                                    BxwXO2  T          @Ǯ@��@�녿B�\��ffB��@��@��\?!G�@�Q�B���                                    BxwX]�  T          @ȣ�@=q@���333��ffB�k�@=q@�  ?(��@��B�u�                                    BxwXl~  T          @�\)@�\@��R�n{�	��B�Q�@�\@���>�G�@���B��R                                    BxwX{$  �          @���@�\@��R�}p���
B���@�\@���>\@e�B�
=                                    BxwX��  �          @��
@
=@�33��(��8(�B���@
=@�\)>�?�  B��                                     BxwX�p  �          @�z�@�R@��\���R�:�RB��@�R@�
==�G�?��
B�u�                                    BxwX�  �          @�z�@:�H@��ÿ����)G�Bt=q@:�H@���>��?�z�Bv\)                                    BxwX��  �          @Å@B�\@��Ϳ�  �<��Bm@B�\@���    =#�
Bp��                                    BxwX�b  �          @\@5@�\)���
�ABv{@5@�(�    ���
Bx�
                                    BxwX�  �          @���?���@�=q�p����HB��?���@�ff�s33��B��                                    BxwX�  �          @�p�?:�H@�
=�H��� =qB�G�?:�H@�=q��\)����B�\                                    BxwX�T  �          @���?�  @�Q��0������B���?�  @��ÿ��R�k�B��                                     BxwX��  �          @���?˅@��Ϳ��H���
B��q?˅@��R�.{�أ�B���                                    BxwY�  �          @���@�@�녿����/33B�
=@�@��>L��?��HB���                                    BxwYF  �          @�G�?�33@�{��{�0��B�(�?�33@�G�>aG�@�B�Ǯ                                    BxwY*�  �          @���?�\@��H��{���HB��3?�\@�=q��{�_\)B�G�                                    BxwY9�  �          @�{?�@������{B��?�@���5���B�#�                                    BxwYH8  �          @�=q?�
=@�G�������{B�G�?�
=@�=q�����{B��
                                    BxwYV�  �          @�=q?�
=@���
=q����B�u�?�
=@��׿J=q��Q�B��                                     BxwYe�  T          @���?޸R@�p��G����\B�z�?޸R@�  �(���ȣ�B�aH                                    BxwYt*  �          @��
?�  @��Ϳ������\B�p�?�  @�(���\)�9��B��q                                    BxwY��  �          @�\)?�
=@��R�����Hz�B�Q�?�
=@��H=���?�=qB���                                    BxwY�v  �          @�=q?xQ�@�(��\(��%G�B�\)?xQ�@�ff>�\)@W�B��R                                    BxwY�  �          @���?�ff@��H�
=��
=B�B�?�ff@��>�
=@��B�ff                                    BxwY��  �          @g���p�@Y��?.{A333B�� ��p�@HQ�?�p�A�33B��=                                    BxwY�h  �          @p�׿��R@QG�?�ffA�p�B�z῞�R@7
=@�
B�HB��f                                    BxwY�  �          @�Q��  @Tz�?�{A�G�B�uÿ�  @<��?��A��B�R                                    BxwYڴ  �          @����&ff@y����R�33B�
=�&ff@|(�>�33@�
=B��H                                    BxwY�Z  �          @�\)���
@}p���=q�k�B�=q���
@�33���
���
B�B�                                    BxwY�   �          @��R��
@u�?z�HAF{B�G���
@^{?��A��HB�W
                                    BxwZ�  �          @����>�R@g
=?}p�A>�RC�\�>�R@P��?�A��CxR                                    BxwZL  �          @�  �N�R@j=q@   A�z�Cp��N�R@C�
@5B=qC�
                                    BxwZ#�  �          @У��w
=@�(�@=qA���C�{�w
=@x��@^�RB�C                                    BxwZ2�  �          @�(���{@�z�?�AG
=C�q��{@�z�@ ��A��C�3                                    BxwZA>  �          @��
��\)@���    <#�
C^���\)@�z�?��A��C	E                                    BxwZO�  �          @��Y��@��Ϳ�33�:�HB�ff�Y��@�������=qB��{                                    BxwZ^�  �          @���fff@�
=�5���HC �f�fff@�  >��@�33C ��                                    BxwZm0  �          @��\�333@�  �����33B���333@����=q�;
=B���                                    BxwZ{�  T          @�Q��5@x���(Q���33B�Q��5@�p���=q��(�B�z�                                    BxwZ�|  �          @�G��0  @���<(����RB��f�0  @�Q������z�B���                                    BxwZ�"  �          @����@��@|(��<�����
B��@��@����������B���                                    BxwZ��  �          @\�P  @�p��@  ��\B��q�P  @�G���{����B�=q                                    BxwZ�n  �          @�{�?\)@p���]p��(�C ��?\)@�Q��=q��ffB�.                                    BxwZ�  �          @��R�G
=@XQ��qG��ffC�f�G
=@�
=�333��
=B��f                                    BxwZӺ  �          @�z��A�@0  �xQ��-�
C	Ǯ�A�@hQ��C�
��C��                                    BxwZ�`  �          @Å�aG�@Z=q�g���HC��aG�@��R�)����(�C�                                    BxwZ�  �          @ۅ�w
=@����}p��Q�C�)�w
=@�p��4z���{C )                                    BxwZ��  �          @����u�@����������Cff�u�@��8Q���=qB���                                    Bxw[R  �          @�Q���33@����n{� {C	���33@���$z���=qCٚ                                    Bxw[�  �          @��H��ff@�  �^{��\)CW
��ff@��������C�                                     Bxw[+�  �          @�
=���H@�z��p  ��Q�C�H���H@�ff�!G���
=C�{                                    Bxw[:D  T          @����ff@�{�xQ���G�C\��ff@����'���Q�C!H                                    Bxw[H�  �          @�\��\)@�ff�����\)C:���\)@���8������C�                                    Bxw[W�  �          @�\)���@�
=�c33��33C
=���@�
=��R��
=Cٚ                                    Bxw[f6  �          @�z���@�(��J=q���CG���@��ÿ��[�C��                                    Bxw[t�  T          A Q���=q@��H�����C	�H��=q@�  �7����C��                                    Bxw[��  
�          A z���{@�����  � �C���{@��H�>�R����C#�                                    Bxw[�(  
�          A���@����������HC�q���@�\)�B�\��=qC�                                    Bxw[��  �          A�R��=q@2�\��{��RCc���=q@~{������p�C��                                    Bxw[�t  T          A���
=@w������
C\��
=@��]p����RCxR                                    Bxw[�  �          AQ���  @n�R������\)C)��  @��R�J�H��Q�C޸                                    Bxw[��  �          Ap���=q@dz������\C����=q@�  �}p���
=C�                                    Bxw[�f  
�          A���p�@s33����ffCJ=��p�@�ff�p������Cn                                    Bxw[�  �          A���33@w���{��C����33@�=q�B�\��\)C�{                                    Bxw[��  T          A�����@w���  ��=qC\)���@����6ff��\)C�                                    Bxw\X  �          A����{@\)��\)�ޣ�C
��{@��R�B�\����CxR                                    Bxw\�  �          A�H��ff@��������33Ck���ff@���]p���{CY�                                    Bxw\$�  �          A(���@�33���
���C���@��H�E���p�C�{                                    Bxw\3J  �          A=q��p�@xQ��p�����RC(���p�@�\)�'
=��
=CW
                                    Bxw\A�  �          A�\��33@�z��`������C���33@����R�V=qC�                                    Bxw\P�  T          A��(�@p  �aG����C���(�@�G��=q�eG�C.                                    Bxw\_<  �          A���
@]p��aG���{C����
@�Q���R�j�\CQ�                                    Bxw\m�  T          A�H�33@Z�H�U���C���33@��z��W�
CY�                                    Bxw\|�  �          A���z�@Mp��<����(�C���z�@xQ��   �<  C+�                                    Bxw\�.  �          A=q�33@G��333����C ���33@p  ��\)�.�HCJ=                                    Bxw\��  T          A
=�=q@@  �!G��j�HC!ٚ�=q@c�
�У��33C��                                    Bxw\�z  T          Aff�
=@8���G��S�
C"��
=@X�ÿ���C��                                    Bxw\�   �          A���(�@*=q����0��C#���(�@C�
������ffC!��                                    Bxw\��  �          A�����@)����Q����C$0����@@  �k����HC"33                                    Bxw\�l  �          A=q���@�
��\)�
=C'�����@ff�E����HC%��                                    Bxw\�  �          A���?�{������RC*W
��?�
=�fff����C(�                                    Bxw\�  �          A�H��Q�@��@�p�AمC�{��Q�@P  @��B
�Cٚ                                    Bxw] ^  �          A���{@���@�z�A�=qC)��{@3�
@�33B33C�                                    Bxw]  �          A
=��G�@�  @��A�ffCff��G�@,(�@�\)B�C�                                    Bxw]�  �          AG���  @�Q�@�B*z�C���  @:=q@�BQCz�                                    Bxw],P  �          A���{@��H@�\)BC�{��{@!G�@�z�B*Q�C�H                                    Bxw]:�  �          A���@�z�@�Q�B��C����@B�\@�=qB.  C�H                                    Bxw]I�  �          A  �+�@�
=@�\)BP(�B잸�+�@4z�A��B��fC�\                                    Bxw]XB  �          A(����
@dz�@��BQ�C�3���
@z�@���B{C"�H                                    Bxw]f�  �          A\)���
@g
=@��B��Cs3���
@z�@�ffB&ffC"�                                    Bxw]u�  T          A�����@mp�@��\B �
Cs3���?�(�@��
B<�C!:�                                    Bxw]�4  �          A{�У�@L(�@�Q�B��C��У�?Ǯ@�B*�RC&��                                    Bxw]��  �          A���z�?�G�@[�A��
C&���z�?B�\@q�A�{C-�3                                    Bxw]��  �          A�H�
�H>�p�?�=qA>=qC1���
�H���?�{AAp�C4��                                    Bxw]�&  �          A���=q>�ff?�  A"ffC0��=q<��
?���A)��C3�)                                    Bxw]��  �          A(��Q�?(�?޸RA>{C/��Q�=�G�?���AIG�C3B�                                    Bxw]�r  �          Aff��=q?�ff@@  A�z�C*E��=q>�
=@O\)A�
=C0�
                                    Bxw]�  �          Aff��\@��@z�A��C#�
��\?�(�@5A�p�C)�                                    Bxw]�  �          A\)�33?�?�AR�RC0G��33���
@   AZ�HC4&f                                    Bxw]�d  �          A���p����
@z�Ae�C4�{�p��&ff?��HAYp�C8��                                    Bxw^
  �          A��
=��@   A�\)C4���
=�Q�@
=A�p�C:
=                                    Bxw^�  �          @�ff���>8Q�@{A��\C2����녿�@�HA��C7�                                    Bxw^%V  �          @����>�(�@�
A��C0�R���u@A���C5��                                    Bxw^3�  T          @��R���>�ff@A{\)C0��������@Q�A�Q�C50�                                    Bxw^B�  �          @�����R>.{@(�A���C2�3��R��ff@	��A�(�C7xR                                    Bxw^QH  �          @�����(�>#�
@\)A�33C2Ǯ��(����@(�A��C8E                                    Bxw^_�  �          @�������>aG�@.{A��
C2G����ÿ\)@*�HA���C8aH                                    Bxw^n�  �          @�\)��z�    @<(�A��C3�R��z�Tz�@4z�A��C:��                                    Bxw^}:  �          @�ff���ͽ�Q�@333A���C4����ͿaG�@*=qA�p�C;J=                                    Bxw^��  �          @���  ��\)@:�HA��C4�\��  �c�
@1�A�C;��                                    Bxw^��  �          @�33���H��@N�RA�
=C5\���H���@Dz�A�p�C<�                                    Bxw^�,  �          @����
��p�@VffA�z�C7:����
���@G
=Ař�C?8R                                    Bxw^��  �          @�Q�����@U�A�Q�C8@ ����z�@C33A���C@\)                                    Bxw^�x  �          @�  �ʏ\�O\)@Z�HA�C;G��ʏ\��G�@C33A��CC�                                     Bxw^�  �          @�ff�ȣ׿s33@Y��A�C<�f�ȣ׿��@>�RA�{CD�                                    Bxw^��  �          @����
=���@aG�A�{C?�R��
=�  @@  A�\)CG޸                                    Bxw^�j  �          @������@p��A�{CA}q����(�@L(�A�ffCJs3                                    Bxw_  �          @�{��\)?�p�@�  B!C ����\)>�Q�@���B/�\C/�
                                    Bxw_�  �          @�Q����@��@�(�B�C�\���?�G�@�z�B5(�C(��                                    Bxw_\  �          @����?��@���B�C&
����#�
@�  B$��C4aH                                    Bxw_-  �          @���=q>Ǯ@z=qB�
C0O\��=q�B�\@vffB ��C;!H                                    Bxw_;�  �          @�����\��\)@}p�B��C4����\��(�@p��B(�C?��                                    Bxw_JN  �          @�����=#�
@��B  C3�=�������R@�\)B�RC@��                                    Bxw_X�  �          @�\)���þu@���B.p�C6�����ÿУ�@���B"{CE�                                    Bxw_g�  �          @ָR��  @<��@�p�B&�RC����  ?\@��HBG�
C(�                                    Bxw_v@  �          @�{���@-p�@�p�B%(�Ck����?��@���BB33C#�                                    Bxw_��  �          @�G����H@.�R@�ffB#�HC�=���H?�ff@��B@��C#G�                                    Bxw_��  T          @ڏ\��z�?ٙ�@�p�B-�\C�H��z�>�  @�
=B;��C0�                                    Bxw_�2  �          @߮��@p�@��B'��C����?�  @��\B@�
C'B�                                    Bxw_��  �          @�(���\)@qG�@e�A�\)C���\)@"�\@���B%\)Cs3                                    Bxw_�~  �          @��
��z�@{@�p�B 33C�R��z�?�ff@�ffB8��C'@                                     Bxw_�$  �          @�(����R@�\@���B(�C.���R?W
=@���B3�C*h�                                    Bxw_��  �          @�����?��
@���B �
C#�����=���@���B+��C2�
                                    Bxw_�p  �          @�{���?\@��B(��C"���=�\)@���B3�RC3=q                                    Bxw_�  �          @���
=?�@��HB&p�C$\��
=���
@��B0(�C45�                                    Bxw`�  �          @�z���
=?��H@��HBz�Cz���
=?�@�\)B-��C-z�                                    Bxw`b  �          @��H��\)?�=q@�G�B  C�
��\)>�ff@���B+��C.Ǯ                                    Bxw`&  "          @�p���z�?���@��\BG�C"�R��z�>aG�@��B'z�C1��                                    Bxw`4�  �          @��H��ff?�p�@�G�BG�C33��ff?��@�{B,�C-&f                                    Bxw`CT  �          @׮��G�@��@�
=BG�C�R��G�?B�\@�p�B/��C*��                                    Bxw`Q�  �          @�ff��\)?��R@�\)B(G�C"�\��\)=u@�
=B3��C3@                                     Bxw``�  �          @�  ��33@�R@���B"�
C���33?J=q@��
B8�
C*5�                                    Bxw`oF  �          @˅���@=q@���BG�CW
���?���@�=qB9(�C%Ǯ                                    Bxw`}�  �          @����]p�@~{@aG�B  C��]p�@-p�@�=qB8=qC�                                    Bxw`��  �          @����|��@XQ�@n{BffC���|��@�@�=qB6CB�                                    Bxw`�8  �          @�{����@E�@|��B��C�{����?�p�@�ffB<{C�=                                    Bxw`��  T          @����\)@   @���B(��C���\)?���@��HBE�C%
=                                    Bxw`��  �          @׮��?�@�
=B2�Cp���>���@��BB�
C0
                                    Bxw`�*  �          @׮��  @��@�(�B"�RC����  ?p��@��B;�\C(=q                                    Bxw`��  �          @���(�@�R@��
B��C��(�?��@�{B2p�C&�R                                    Bxw`�v  �          @���@�R@�=qB�C{��?���@�z�B/�
C&                                    Bxw`�  T          @ָR��@�\@��RBC�H��?aG�@��RB2�HC)^�                                    Bxwa�  �          @���p�?޸R@�B2ffC����p�>L��@��BA��C1n                                    Bxwah  �          @���?��@�=qB8��C#)���aG�@�  BA�RC6ٚ                                    Bxwa  
�          @�ff����?�
=@��B7�C$���������
@�  B?�C88R                                    Bxwa-�  �          @ʏ\����?&ff@��B;  C+W
���ÿQ�@��\B9��C>�\                                    Bxwa<Z  �          @�(���\)?fff@�ffB=�\C(  ��\)��R@��B?��C<G�                                    BxwaK   �          @�=q��Q�?��@�=qB8�C%����Q����@�{B>z�C9O\                                    BxwaY�  "          @�{�~�R?˅@���BA
=C.�~�R    @��BOQ�C3�                                    BxwahL  �          @˅�|(�?�p�@�B=C=q�|(�>.{@��BN�
C1��                                    Bxwav�  �          @����{?��\@�G�B9C&Y���{��@�(�B>�\C:Q�                                    Bxwa��  �          @˅���?��
@�ffB?�
C%�H����   @�G�BD�C;                                      Bxwa�>  "          @�z����?xQ�@�G�BBz�C&�R������@�33BE��C<E                                    Bxwa��  �          @�p����?���@�{B/p�C"�f�������@���B9��C5T{                                    Bxwa��  �          @����\)?�@�(�B0G�C���\)=�G�@�p�B>�RC2��                                    Bxwa�0  �          @љ�����?�=q@��
B'=qC�����>�{@�
=B8p�C/��                                    Bxwa��  �          @љ���\)?�p�@��RB+��C޸��\)>aG�@���B;\)C1=q                                    Bxwa�|  �          @����33?�33@��B&=qC����33>Ǯ@�G�B8�C/5�                                    Bxwa�"  �          @أ���(�@(�@��B��C�=��(�?=p�@���B-�C+W
                                    Bxwa��  �          @������@�@u�B�C�����?G�@��B)ffC*��                                    Bxwb	n  �          @�����G�@$z�@�z�B��C}q��G�?���@�  B5�RC&\)                                    Bxwb  �          @޸R���R@K�@�G�B"�
C����R?�ff@�=qBG�\C��                                    Bxwb&�  T          @޸R�|��@U@���B'��C���|��?�z�@�\)BO�C@                                     Bxwb5`  �          @�\)�o\)@Mp�@�{B4Q�CW
�o\)?�Q�@��RB[�HC�R                                    BxwbD  �          @ۅ�z�H@G�@�B,Q�C���z�H?�
=@�{BR{C                                       BxwbR�  �          @ٙ��vff@E�@�{B.p�CY��vff?���@�{BT=qC 8R                                    BxwbaR  �          @�(��{�@>{@�G�B0��C޸�{�?�  @�  BT�\C"W
                                    Bxwbo�  �          @�=q�z�H@O\)@��
B(�Ch��z�H?�(�@�ffBD��CY�                                    Bxwb~�  �          @ҏ\��Q�@U@\)B�C&f��Q�?���@��
B?(�C+�                                    Bxwb�D  �          @�z��s�
@J�H@���BffCG��s�
?�@�33BEG�CQ�                                    Bxwb��  �          @��h��@B�\@��\B)p�C&f�h��?���@��HBQz�CJ=                                    Bxwb��  �          @ҏ\��z�@K�@\)B�CxR��z�?�Q�@��B<p�C��                                    Bxwb�6  �          @أ����\@i��@k�B�C�����\@�R@�B/ffC                                    Bxwb��  �          @љ����\@Vff@uB(�C�f���\?��@�\)B9��C(�                                    Bxwbւ  �          @�����@W
=@���B�\C�����?�@��B=�HCٚ                                    Bxwb�(  �          @θR��@��@��RB$C.��?\(�@���B@��C(\)                                    Bxwb��  �          @��H����?��R@�\)B+p�C������>Ǯ@�(�B?��C.�q                                    Bxwct  �          @��H���H@  @�33B  C�H���H?B�\@��
B333C*��                                    Bxwc  �          @�\)��{@p�@~{B{C���{?}p�@�=qB4Q�C'aH                                    Bxwc�  �          @�33��  @ ��@�  B(�C�R��  >��@�B>G�C-�3                                    Bxwc.f  �          @�\)���@9��@o\)B�\CxR���?�p�@�  B8��C �                                    Bxwc=  �          @��H�Q�@�@�\)BQ  C@ �Q�>u@��
Bj(�C/�{                                    BxwcK�  T          @��
�P  ?У�@�ffB\
=Ck��P  �aG�@�{BlG�C7��                                    BxwcZX  �          @�33�!G�?���@��B}33C=q�!G��(��@���B�u�CB��                                    Bxwch�  �          @��H�   ?8Q�@�ffB��C 
�   ��=q@��HB�W
CU��                                    Bxwcw�  �          @���8��?��@�=qBq�\C�H�8�ÿ+�@�By�CA�                                    Bxwc�J  �          @����(Q�?���@�G�B{\)C�{�(Q�^�R@�=qB~=qCFL�                                    Bxwc��  �          @Ϯ�S33?�  @�  BZ(�C#��S33�\)@���Blp�C6s3                                    Bxwc��  �          @У��n�R@ff@���B=C�q�n�R?z�@�=qBZ�\C+33                                    Bxwc�<  �          @�33�XQ�@�@��BN
=C^��XQ�>�(�@���Bk��C,��                                    Bxwc��  �          @�z��|(�@�R@�\)B5�HC޸�|(�?5@���BS�\C)�\                                    Bxwcψ  �          @�33�l(�@(��@���B:  Cff�l(�?W
=@���B[�
C'!H                                    Bxwc�.  �          @�(��j=q@;�@�{B4
=C^��j=q?��@��BZ��C"��                                    Bxwc��  �          @Ӆ�{�@5�@��B+33CJ=�{�?�{@�{BO  C$8R                                    Bxwc�z  �          @ۅ�y��@_\)@�B!
=C
33�y��?�  @��BMC�H                                    Bxwd
   
�          @��
�vff@w
=@�p�B�C���vff@(�@��BGG�Ck�                                    Bxwd�  �          @���mp�@��@y��B�HC�R�mp�@#33@�33BB(�Cu�                                    Bxwd'l  �          @����g�@��R@j�HB�
C�3�g�@*�H@���B=C��                                    Bxwd6  �          @�ff�c33@�  @S�
A���C��c33@5�@��\B4�HCs3                                    BxwdD�  �          @�Q��j=q@�ff@A�A�=qCz��j=q@G
=@�z�B)�RC�f                                    BxwdS^  �          @���c�
@�{@-p�A�{C � �c�
@Mp�@��HB!�HC	�3                                    Bxwdb  �          @�ff�[�@�ff@$z�Aģ�B�=q�[�@QG�@~{B��CY�                                    Bxwdp�  �          @ƸR�b�\@�G�@0��A��HC}q�b�\@C�
@��HB$��C.                                    BxwdP  �          @����q�@�z�@-p�A�ffC��q�@J=q@��\Bp�C
=                                    Bxwd��  �          @��
����@�p�@
=qA���CY�����@XQ�@e�B�
C�                                    Bxwd��  �          @��H�~�R@�Q�?޸RA��RC��~�R@XQ�@HQ�A�p�C��                                    Bxwd�B  �          @�33�~�R@�G�?�
=A�{C���~�R@Z�H@EA�CB�                                    Bxwd��  �          @�����=q@�
=?�=qALQ�C�3��=q@^�R@/\)A�z�CxR                                    BxwdȎ  T          @����QG�@���>��@�G�B��f�QG�@u�?��HA��C�=                                    Bxwd�4  �          @��?���@#33��p��i�\Bz�R?���@����]p��z�B��H                                    Bxwd��  �          @���?��
?������R�RB8G�?��
@j�H��ff�>  B�B�                                    Bxwd�  �          @��?xQ�@AG���=q�W  B�8R?xQ�@���=p��G�B���                                    Bxwe&  �          @��ý�\)@s33�R�\�#�B�33��\)@�
=���H���B�                                    Bxwe�  �          @���G�@���)����Bɨ��G�@�  ���\�@z�Bƙ�                                    Bxwe r  �          @�  �(��@|(��p���2�RB���(��@���?
=q@�z�B��3                                    Bxwe/  �          @�{�W
=@qG�>�z�@QG�C�W
=@Z=q?�\)A���C�{                                    Bxwe=�  T          @�p����@�=q��  ��33B޳3���@�G�>�{@dz�B�                                    BxweLd  �          @��^�R@�p���Q쿆ffB�
=�^�R@��
?�Q�A��B�L�                                    Bxwe[
  �          @�33�z�H@�Q��\)���\B�(��z�H@��>Ǯ@�Q�B�G�                                    Bxwei�  �          @�33����@�  ��ff���\B˸R����@��R>�@�
=B��)                                    BxwexV  �          @�zΰ�
@�Q�h��� (�BЮ���
@�  ?xQ�A)p�BиR                                    Bxwe��  �          @�p��:�H@c�
@/\)A�  CaH�:�H@z�@vffB7p�C��                                    Bxwe��  �          @��\���R@��H>�@�
=B��
���R@��@ffA�\)B��                                    Bxwe�H  �          @���>��@tz��i���.�
B�\)>��@�z�� �����RB�aH                                    Bxwe��  �          @��
�J=q@�{�P����\B�ff�J=q@��H��  ��(�BŮ                                    Bxwe��  �          @���?&ff@�G��?\)�
33B��f?&ff@��H���H�Tz�B��=                                    Bxwe�:  �          @�(�?�R@����U����B�ff?�R@�
=��\)��p�B��3                                    Bxwe��  �          @�ff@�
@\)�����\�
BI��@�
@����Z=q��B{�R                                    Bxwe�  �          @�p�?���@\(���Q��?�\B��R?���@��p��أ�B��H                                    Bxwe�,  �          @�{?���@l���s33�2�\B��)?���@��H�
�H���B�(�                                    Bxwf
�  �          @�
=?k�@�ff�W��ffB�k�?k�@��Ϳ�=q���\B��)                                    Bxwfx  �          @�ff?0��@���N�R��
B��R?0��@�ff��33�o\)B���                                    Bxwf(  }          @�G�?#�
@���:�H�Q�B�B�?#�
@�(������@(�B��q                                   Bxwf6�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxwfEj  	/          @��=#�
@�p��J=q��B��=#�
@��������t��B��                                    BxwfT  �          @��=u@��%��  B��{=u@�녿(�����
B�                                    Bxwfb�  �          @�z��@��
�޸R����B�33��@�=�?���B�                                      Bxwfq\  �          @���>�R@�(�?�G�A,��B�aH�>�R@l(�@%�A�ffC �                                    Bxwf�  �          @�\)�e�@�{?ǮA|(�Cs3�e�@S33@@��A�{C	@                                     Bxwf��  �          @�(��vff@Z=q@   A��C
xR�vff@=q@HQ�B
=qC�                                    Bxwf�N  �          @�(��y��@:=q@  A���CJ=�y��?���@K�B��C��                                    Bxwf��  �          @�{�~�R@:=q@33A�  Cٚ�~�R?�=q@N{Bp�CW
                                    Bxwf��  �          @�  �tz�@7�@�A�Q�C!H�tz�?�=q@FffB�
Cff                                    Bxwf�@  �          @�Q��s�
@6ff@\)A�G�C8R�s�
?��@I��B{C��                                    Bxwf��  �          @����k�@<��@
=A�z�CE�k�?�@S33B��Cu�                                    Bxwf�  T          @�G��j�H@AG�@z�A�ffC���j�H?�z�@S33B33Cp�                                    Bxwf�2  �          @����p��@@  @
�HA�{C\)�p��?���@I��Bz�C�                                    Bxwg�  �          @�33�qG�@>{@A�Q�C�R�qG�?���@S33B��CǮ                                    Bxwg~  �          @�(��g
=@@��@'
=A�(�C0��g
=?��@c�
B%�
C�                                    Bxwg!$  �          @���Y��@N�R@+�A��
Cn�Y��?��H@mp�B-��C��                                    Bxwg/�  �          @�\)�U�@dz�@�RA׮C\�U�@@j=qB'�RC��                                    Bxwg>p  �          @����]p�@X��@*=qA��C���]p�@
=@p  B+(�C��                                    BxwgM  �          @��\�Z=q@o\)@�A�p�CaH�Z=q@#�
@eB 33C#�                                    Bxwg[�  �          @����N�R@��@��A��C (��N�R@>{@c�
B{C	c�                                    Bxwgjb  �          @���G�@�33?�=qA�
=B�B��G�@S�
@W�B�CO\                                    Bxwgy  �          @����7
=@���?��
A�G�B� �7
=@_\)@Y��B�HCT{                                    Bxwg��  �          @���!G�@���?�33A�p�B�u��!G�@q�@XQ�BffB�\)                                    Bxwg�T  �          @��>�R@��
?�z�Af�\B��\�>�R@mp�@EB(�C �R                                    Bxwg��  �          @�{�mp�@��\?�G�AMCB��mp�@QG�@0  A�=qC
�{                                    Bxwg��  �          @�=q�N{@��H?�ffAYG�B�{�N{@_\)@8��A��RC��                                    Bxwg�F  �          @�G��qG�@w�?��HAIp�CB��qG�@E@'�A��C��                                    Bxwg��  �          @�{��@o\)?n{AffC
8R��@E�@�
A�{C��                                    Bxwgߒ  �          @��R��Q�@g�?Q�A�C�
��Q�@@��@
�HA���C��                                    Bxwg�8  �          @�����
@2�\@�\A��
C����
?�@L��B  C�R                                    Bxwg��  �          @���y��@HQ�@��A¸RC8R�y��?��R@S�
B\)C�                                    Bxwh�  �          @�
=�w�@K�@��A�G�C�
�w�@�@N{B�C�R                                    Bxwh*  �          @���{�@[�?�Q�A�\)C
��{�@\)@9��A��C�3                                    Bxwh(�  �          @�(��z�H@]p�?�p�AS�
C
���z�H@,(�@\)A�G�C�\                                    Bxwh7v  �          @����z�H@dz�?��A1��C	���z�H@7�@�A��
C��                                    BxwhF  �          @��R�~{@j=q?E�A{C	O\�~{@C�
@
=qA�ffCff                                    BxwhT�  �          @�\)�}p�@mp�?333@�=qC��}p�@HQ�@
=A�p�C�q                                    Bxwhch  �          @���w�@w
=>��@���C��w�@XQ�?�z�A��C
��                                    Bxwhr  �          @�
=���\@g�<�>��
C
ff���\@S�
?�(�Ay�C�                                    Bxwh��  �          @�{��G�@h�ü��
�8Q�C	����G�@U?�
=At��CaH                                    Bxwh�Z  �          @�{�~{@j�H���
�\(�C	J=�~{@_\)?�AG
=C
�                                    Bxwh�   �          @�����\@[��   ��{C�����\@U?h��A\)CQ�                                    Bxwh��  �          @�33����@e������HC�����@^�R?�G�A%p�C��                                    Bxwh�L  �          @�����\@i������33Ch����\@X��?�{AVffCs3                                    Bxwh��  �          @�����\@]p��h���\)C�R���\@a�?\)@���Cff                                    Bxwhؘ  �          @�\)���@p�׿�z��eG�C	5����@�Q�>aG�@��C\)                                    Bxwh�>  �          @��~{@qG����H�n�\CxR�~{@�G�>B�\?��C�                                     Bxwh��  �          @�z����@s�
�n{�z�Cٚ���@w
=?.{@�{Cs3                                    Bxwi�  �          @�����@h�ÿ��
�+33C
+����@o\)?�@�
=C	Y�                                    Bxwi0  �          @�����\)@`�׿8Q���p�C^���\)@_\)?E�@�{Cu�                                    Bxwi!�  �          @�G����@k��\�z=qC
+����@aG�?�33A?�Cu�                                    Bxwi0|  �          @����{�@b�\���R�VffC	�R�{�@W
=?�z�AICs3                                    Bxwi?"  �          @��tz�@\(��Ǯ��33C	�R�tz�@S33?��A:�HC#�                                    BxwiM�  �          @��
�`  @n{�u��RCB��`  @Z�H?�(�A�p�C�                                    Bxwi\n  �          @�(��P��@mp�����B{CO\�P��@tz�?��@�ffCz�                                    Bxwik  �          @�(��  @x�������ffB�{�  @����#�
��B螸                                    Bxwiy�  �          @��R���
@�������B݊=���
@���   ��(�B�k�                                    Bxwi�`  �          @�p�����@\)��R��33B�(�����@����Ǯ��=qB߳3                                    Bxwi�  �          @���
=@|(��   ����B�Q��
=@���L�����B�{                                    Bxwi��  �          @�=q����@tz��G���\)B�����@�p��   ��ffB�                                    Bxwi�R  �          @�=q��z�@h���*=q�ffB���z�@�p��k��0��B�(�                                    Bxwi��  �          @�{��@.{�e��:
=B��\��@|�������\)B�ff                                    Bxwiў  �          @�p���\@e�!G����RB�����\@�녿O\)�{B��f                                    Bxwi�D  T          @����'�@o\)��33��Q�B���'�@\)>�\)@S�
B��\                                    Bxwi��  �          @�\)�$z�@j�H��  ���B���$z�@}p�>#�
?�B�{                                    Bxwi��  �          @�
=�z�@r�\�˅��
=B����z�@��>�?�{B��                                    Bxwj6  �          @�=q���@�����ff���B�  ���@�=L��?��B�(�                                    Bxwj�  �          @�  ��G�@�����
�Ù�B��
��G�@�  ��Q쿆ffB�B�                                    Bxwj)�  �          @��Ϳ.{@w
=�S33� z�B��H�.{@�z῱��{�B�                                    Bxwj8(  �          @�  ��p�@p  �h���033B�\)��p�@��޸R��p�B���                                    BxwjF�  �          @�=q�Ǯ@�
=�K���B���Ǯ@����\)�C\)B��R                                    BxwjUt  �          @������@~�R�]p��#33B�����@�=q��(����B���                                    Bxwjd  �          @�녿�@g
=�vff�:{B�p���@��Ϳ�p�����B�(�                                    Bxwjr�  �          @��ÿ˅@�(��3�
���B�=q�˅@��O\)��RB��)                                    Bxwj�f  �          @�33��Q�@�33��(���  Bس3��Q�@���=L��?�Bգ�                                    Bxwj�  �          @�
=��(�@�=q�C33��RB�uÿ�(�@�\)��ff�9G�BϞ�                                    Bxwj��  T          @��\��ff@H���o\)�A�B�\��ff@�p���
��33B�                                    Bxwj�X  �          @���R@O\)����q�B�\��R@XQ�>��@�  B��R                                    Bxwj��  �          @�Q��<��@*=q?�33A�33C	޸�<��?���@��B
�RCG�                                    Bxwjʤ  �          @���G�@(Q�?�{AtQ�C���G�?�z�@�A��C�\                                    Bxwj�J  �          @�
=�'�@J�H?n{ALz�C�\�'�@�R@(�A�z�C��                                    Bxwj��  �          @�G��(Q�@I��?�(�A�=qCٚ�(Q�@�@(�B33C
k�                                    Bxwj��  T          @�=q�#�
@R�\?��Adz�B��q�#�
@!�@
=B=qCc�                                    Bxwk<  �          @�G��	��@l(�>L��@*=qB��	��@P  ?�  A��B�{                                    Bxwk�  �          @���=q@^{?W
=A333B��3�=q@1�@\)A�z�C                                      Bxwk"�  �          @�=q��@:�H?�G�A�(�C ����?��R@'
=B!�Ck�                                    Bxwk1.  �          @|����@"�\?�A�z�C�q��?�G�@0  B1\)Cz�                                    Bxwk?�  �          @tz����@:�H?�=qA��B������@�@{B�
CO\                                    BxwkNz  �          @s33�z�@6ff?}p�Aq�C!H�z�@��@�B�C	5�                                    Bxwk]   �          @�=q�1�@)��?k�AXz�C\)�1�?��R?�(�A�  CT{                                    Bxwkk�  �          @�{�'
=@J=q?G�A-�C���'
=@ ��@z�A��C&f                                    Bxwkzl  �          @�=q�"�\@C�
?0��A�\C���"�\@p�?���A�RC�3                                    Bxwk�  �          @u��%@2�\>\@��C�=�%@
=?�ffA���C	��                                    Bxwk��  �          @~{� ��@B�\>�33@�z�C�� ��@%?��A�\)C�                                    Bxwk�^  �          @tz��33@B�\>�z�@��
B�\)�33@'
=?�=qA�33Ck�                                    Bxwk�  �          @s�
�#�
@333>8Q�@0  Cs3�#�
@��?���A�Q�CY�                                    Bxwkê  �          @j�H��@3�
>�?��RC���@{?�=qA���CG�                                    Bxwk�P  �          @e��@0  >u@w
=C����@�?�A�{C޸                                    Bxwk��  �          @xQ���R@@  =L��?:�HC�
��R@+�?���A���C�\                                    Bxwk�  �          @��\�G�@W
=>���@��B���G�@5?�=qA�33C ��                                    Bxwk�B  T          @����@X��>\@��B��)��@8Q�?�=qA��HCh�                                    Bxwl�  �          @����@h��=�Q�?�33B����@O\)?�A�
=B�p�                                    Bxwl�  �          @���)��@Fff?&ffAG�C���)��@   ?�Q�A���C�f                                    Bxwl*4  �          @�  �P��@h��?�  A9Cٚ�P��@5�@   A��\C�                                    Bxwl8�  �          @�\)�l(�@Tz�?��A�{C

=�l(�@��@;�B\)C�                                     BxwlG�  �          @�G��`  @3�
@	��A��C:��`  ?�=q@J�HBG�C                                    BxwlV&  �          @����(�@�@�
A�(�C����(�?�z�@H��B{C$J=                                    Bxwld�  �          @������@z�@{A��C8R����?B�\@HQ�B{C)�H                                    Bxwlsr  �          @�����33?�\)@�HA�=qC���33?��@@  B�C,��                                    Bxwl�  �          @�����?��@#�
A�RC$
=���=#�
@7�B ��C3��                                    Bxwl��  �          @��H��(�?�@�A�G�C,�H��(��
=@ffA�p�C<
                                    Bxwl�d  �          @�����>�G�@&ffA��C.
=���B�\@!G�A�G�C>L�                                    Bxwl�
  �          @�z�����?��R@N{BC%u����׾��
@[�B�HC7޸                                    Bxwl��  �          @����=q?���@K�B Q�C"  ��=q�#�
@b�\B=qC4�                                    Bxwl�V  �          @��R��G�?У�@I��A��C!0���G�=u@c33B
=C3L�                                    Bxwl��  T          @������@�\@N�RBz�Cc����>Ǯ@s33B33C/B�                                    Bxwl�  
�          @�  ��@\)@>�RA�33Cu���?+�@j�HB33C+�)                                    Bxwl�H  �          @�
=��  ?��R@:=qA��HC&f��  >�@_\)B��C-�                                    Bxwm�  �          @�����ff@ ��@4z�A�(�C���ff?
=q@[�Bz�C-#�                                    Bxwm�  �          @��
��Q�@�R@
=AѮCff��Q�?n{@G
=B�C'�                                    Bxwm#:  �          @��H��G�@
=@��Aԏ\Cٚ��G�?O\)@E�B�
C)O\                                    Bxwm1�  T          @����HQ�@@��?���A���C)�HQ�?�{@@  BG�CO\                                    Bxwm@�  "          @����ff@*�H@��A��C���ff?�@G�B�
C"Q�                                    BxwmO,  "          @�{����@�R@
=qA�{C�����?��\@;�A��C'�                                    Bxwm]�  
�          @�p���G�@�@\)A�(�C
=��G�?L��@:=qA���C*}q                                    Bxwmlx  "          @����\)@��@��A�p�C���\)?aG�@HQ�B\)C)h�                                    Bxwm{  
Z          @�=q��
=?�z�@A�\)C�q��
=?&ff@<��A�p�C,�{                                    Bxwm��  
�          @��\��Q�?���@�RA�=qC�R��Q�?:�H@7�A�z�C+�3                                    Bxwm�j  
�          @�=q����?�Q�@��A��C�
����?=p�@5A��
C+�f                                    Bxwm�  
�          @��
���?�(�@�A��C!8R���>�(�@<��A�33C/(�                                    Bxwm��  �          @�=q��\)?�(�@�RA�p�C ����\)>���@?\)A���C/Y�                                    Bxwm�\  
�          @����33?���@Q�A�{C"����33>�{@6ffA�
=C0&f                                    Bxwm�  �          @�{���@�@\)AхC@ ���?@  @L(�B	Q�C*�3                                    Bxwm�  �          @�����33@G�@%�A�=qCp���33?W
=@U�B�HC)�                                    Bxwm�N  	�          @�����H@
=q@-p�A陚C�����H?0��@Y��B�
C+�                                    Bxwm��  
Z          @�\)��Q�?��@{A�=qC:���Q�?\)@C�
B�
C,�R                                    Bxwn�  "          @�  ��
=@�R@33A�  C� ��
=?n{@Dz�B�\C(B�                                    Bxwn@  �          @��R��  @$z�?ٙ�A�33C:���  ?�G�@,(�A�=qC!s3                                    Bxwn*�  
(          @�\)��Q�@33@�A�33C  ��Q�?�{@7
=A��C&.                                    Bxwn9�  
(          @�����R@�?�z�A��HC� ���R?�33@%�A�  C#�                                     BxwnH2  
�          @�����z�@\)?�G�A��
C���z�?�33@-p�A�C#+�                                    BxwnV�  "          @������H@!�?�\A��C����H?�
=@/\)A�p�C"�                                    Bxwne~  "          @�  ���@ff?���A�Q�C�3���?�Q�@3�
A��RC%Y�                                    Bxwnt$  	�          @�����@�?�ffA�33C@ ���?xQ�@!�A�  C()                                    Bxwn��  T          @�����(�?�?�G�A�ffCz���(�?c�
@��A�=qC)�)                                    Bxwn�p  �          @��R��{?ٙ�?��A�\)C!�{��{?@  @\)A��HC+�R                                    Bxwn�  �          @�  ���?�p�@�
A�  C$����>�33@ ��A���C0�                                    Bxwn��  �          @������?�33@	��A�z�C'�{���<�@(�AǙ�C3��                                    Bxwn�b  
�          @�=q����?�
=@�AŮC'����ͽ��
@,��A�\)C4��                                    Bxwn�  
�          @������?�=q@(��A֏\C%5������#�
@=p�A��HC4��                                    Bxwnڮ  �          @�  ��(�?\@(Q�Aٙ�C"���(�>�@A�A��HC2��                                    Bxwn�T  �          @�{���?�p�@,(�A�p�C�q���?   @S33B=qC.�                                    Bxwn��  "          @�  ���R?�p�@8��A�Q�C ���R>B�\@W
=BffC1�{                                    Bxwo�  �          @�������?�G�@?\)A�C������>#�
@]p�B��C2                                    BxwoF  "          @�=q��
=?�@<��A�RC����
=>�  @^{BffC1!H                                    Bxwo#�  
�          @�  ����?�(�@3�
A��
C!H����>�(�@Y��B
��C/                                      Bxwo2�  "          @����p�@��@+�A؏\C����p�?^�R@_\)BffC)}q                                    BxwoA8  
�          @���z�@�@@��A���C}q��z�>�@i��BQ�C.Y�                                    BxwoO�  �          @�33��Q�@{@?\)A��C� ��Q�?\)@k�B�C,�H                                    Bxwo^�  
�          @�G���@�\@1G�A�(�CxR��?   @Y��B�\C-�                                    Bxwom*  �          @�=q����?��@1�A�RCxR����>�=q@S33B
33C0��                                    Bxwo{�  �          @��H���R?�ff@   Aʏ\C ����R>�
=@C33A�{C/0�                                    Bxwo�v  
�          @������?���@'�Aԣ�C$�������#�
@=p�A�G�C4�                                    Bxwo�  
�          @�z���ff?�G�@4z�A�RC#���ff    @L��B�C3��                                    Bxwo��  �          @��R���?���@9��A�
=C#������
@O\)B��C4�R                                    Bxwo�h  
�          @�{��?�{@A�A��C$�)���k�@S�
Bp�C6��                                    Bxwo�  "          @\��{?��@R�\B=qC%h���{��
=@`��B�C8�H                                    BxwoӴ  �          @�  ��z�?�p�@N{B\)C%ٚ��z��G�@Z�HB�\C9!H                                    Bxwo�Z  
�          @�  ��z�?��@P  B�C'n��z�z�@XQ�B	�
C:�                                     Bxwo�   �          @�(���  ?�\)@1G�A���C$����  ����@EA���C5+�                                    Bxwo��  �          @�(���  ?��R@4z�A�C&!H��  ��  @Dz�A��C6�)                                    BxwpL  T          @�{��z�?�G�@HQ�B \)C(J=��z�
=@O\)B=qC:�                                    Bxwp�  
�          @�  ���?
=q@X��B	�C-���������@O\)B
=CA�                                    Bxwp+�  T          @�Q����>�\)@aG�B�\C0�������
=@N�RB33CD\)                                    Bxwp:>  
�          @�G���(�>��@q�B#Q�C2��(���@Y��B�HCH�)                                    BxwpH�  �          @�33��=��
@Z=qBQ�C2�����Ǯ@A�B�CGY�                                    BxwpW�  �          @�\)��>�33@QG�B�RC/������  @C33B ��CB�R                                    Bxwpf0  "          @�\)��=q?���@"�\A��C!����=q>.{@>�RA��C2
=                                    Bxwpt�  
(          @�p���
=?��@3�
A�  C#�{��
=��@H��B\)C5z�                                    Bxwp�|  "          @��
��(�?�G�@<(�A�C$��(�����@K�B	��C7�                                    Bxwp�"  
�          @��
��ff?�G�@*=qA��C"33��ff=L��@C�
B33C3aH                                    Bxwp��  "          @��H��ff?�G�@#�
A�G�C"(���ff=���@>{B=qC2Ǯ                                    Bxwp�n  "          @�����p�?���@=p�A�G�C&n��p���(�@H��B=qC9J=                                    Bxwp�  
�          @�������>\@e�BC/  �������@Tz�BQ�CE��                                    Bxwp̺  "          @�����?�=q@AG�B(�C&�3�����\@J�HBG�C:0�                                    Bxwp�`  T          @������?s33@?\)B �C(� ���Ϳ(�@Dz�B�C;k�                                    Bxwp�  �          @���{?c�
@Dz�BC)@ ��{�0��@G�B(�C<k�                                    Bxwp��  
�          @�
=���
?n{@O\)B
�C(�)���
�@  @R�\Bp�C=.                                    BxwqR  "          @�Q����?xQ�@O\)B��C(0�����5@S�
BG�C<�f                                    Bxwq�  �          @�����33?�G�@E�B��C'�\��33���@K�B	�HC;c�                                    Bxwq$�  T          @�����R?��@'�A�\)C$�3���R��@:�HA�C5��                                    Bxwq3D  
�          @�=q����?��R@?\)B(�C$�3���׾�Q�@N{BQ�C8��                                    BxwqA�  
�          @������H?�Q�@333A�=qC%k����H���R@A�BQ�C7�
                                    BxwqP�  T          @��\����?��@'
=A�p�C'33�������R@3�
A�z�C7��                                    Bxwq_6  "          @�����?�z�@*=qA�(�C&�������\)@8��A�G�C7B�                                    Bxwqm�  "          @��R���?�  @-p�A�
=C(u������G�@7
=A�z�C9{                                    Bxwq|�  T          @��\���?��H@1�A��C%p������z�@AG�Bz�C7�{                                    Bxwq�(  "          @�\)��\)?�=q@5A���C#}q��\)�aG�@HQ�B�C6�{                                    Bxwq��  
Z          @�������>Ǯ@,(�A�C/c������xQ�@"�\A�33C?n                                    Bxwq�t  �          @�
=��=q>#�
@{A�\)C2)��=q����@\)A�33C@xR                                    Bxwq�  �          @������?k�@=qA�p�C)������p�@#33A��C8!H                                    Bxwq��  T          @�z����?n{@ ��Aə�C)��������@(��A�G�C8z�                                    Bxwq�f  �          @����\)?h��@z�A���C*{��\)����@{A��C7�f                                    Bxwq�  �          @�ff��z�?s33@\)A��C)}q��zᾅ�@�HA��
C6�                                    Bxwq�  �          @�Q�����?:�H@(��A��HC+�{�����(��@*=qA�ffC;k�                                    Bxwr X  �          @�Q����H?Tz�@$z�A�ffC*�����H��@)��A�\)C9�q                                    Bxwr�  �          @���Q�?��
@{A�{C(\)��Q쾞�R@*=qA�33C7��                                    Bxwr�  �          @�����?�{@!G�A�=qC'\)����=q@/\)A�Q�C7&f                                    Bxwr,J  �          @�\)����?��@\)A�z�C(���������@,(�A�z�C7^�                                    Bxwr:�  �          @�Q���33?h��@ ��AθRC)����33��G�@(Q�A�C8�                                    BxwrI�  �          @�\)���H?���@Q�Aģ�C'�����H�W
=@'
=A�\)C6aH                                    BxwrX<  �          @�{��
=?W
=@)��A�Q�C*W
��
=��@.�RA���C:��                                    Bxwrf�  �          @��H��\)?s33@3�
A��C(�H��\)���@:�HA��
C:��                                    Bxwru�  
�          @�ff���H?xQ�@6ffA�C(�R���H�\)@<��A��C:�
                                    Bxwr�.  �          @�  ���?c�
@@��A���C)z���녿333@C�
B C<T{                                    Bxwr��  �          @�
=��
=?s33@Dz�B{C(����
=�0��@H��BQ�C<L�                                    Bxwr�z  �          @�{���?Tz�@:=qA�Q�C*5���녿5@<(�A�33C<k�                                    Bxwr�   �          @�p���
=?��@(��A�ffC-����
=�Q�@$z�A�C=h�                                    Bxwr��  �          @��R����?��@$z�A���C-������J=q@ ��A��C<�f                                    Bxwr�l  �          @�ff���
?�  @G�A�Q�C(�����
���@{AͮC6޸                                    Bxwr�  �          @���\)?�  @Q�A���C%���\)��Q�@,(�A�\C5
=                                    Bxwr�  �          @��\��
=?�
=@A��
C#�R��
=>W
=@!G�A���C1�)                                    Bxwr�^  �          @��H���?��@ ��A�
=C�����?!G�@+�A�33C,��                                    Bxws  �          @�=q��?�ff?�\)A��RC���?#�
@ ��A֣�C,�)                                    Bxws�  �          @�����
=?��H?��
A��C �q��
=?��@��A�z�C-�                                    Bxws%P  �          @�����?\@�A��C"�{��>��
@!G�A��
C0T{                                    Bxws3�  �          @�=q���R?���@  A��C'5����R�\)@!G�A���C5�H                                    BxwsB�  �          @��\��ff?�=q@(�A�
=C$����ff=�\)@#�
AڸRC3@                                     BxwsQB  T          @�{���?�G�@z�A���C%u����=u@�A�=qC3E                                    Bxws_�  �          @�z���G�?z�H@�A�  C(u���G���z�@p�A�p�C7h�                                    Bxwsn�  �          @�����
=?���@�
A��C&ff��
=<��
@Q�A�33C3��                                    Bxws}4  �          @��H����?���?��HA���C%������>�z�@��A�=qC0�=                                    Bxws��  �          @�����z�?��R?�(�A��RC#���z�>��R@��Aԣ�C0aH                                    Bxws��  �          @��
��=q?���@'
=Aݙ�C&���=q���@7
=A��C7
                                    Bxws�&  �          @�p�����?�p�@!�A��HC%����;L��@333A�(�C6Q�                                    Bxws��  �          @������?�\)@:�HA���C&n���;�@EB�RC9�                                    Bxws�r  �          @�p����\?�@+�A�Q�C&W
���\���
@9��A��C7�=                                    Bxws�  �          @�z���G�?���@ ��A�(�C#:���G�>�p�@"�\A̸RC/��                                    Bxws�  �          @���
=?˅@33A��HC#���
=>aG�@2�\A��\C1�=                                    Bxws�d  �          @�����R?���@'�A��
C%p����R�\)@<��A��C5�\                                    Bxwt
  �          @�Q����?�
=@$z�A�(�C$������L��@<(�A�{C4��                                    Bxwt�  �          @�����\)?�Q�@'�A�{C$����\)��\)@?\)A�C4�q                                    BxwtV  �          @�
=��{?�{@.{A���C'�H��{����@:=qA陚C8c�                                    Bxwt,�  �          @�33��ff?���@9��A��HC%p���ff��z�@K�A�C7(�                                    Bxwt;�  �          @����ff?���@?\)A癚C$ff��ff�u@S�
B�
C6�                                    BxwtJH  �          @�  ��33?p��@X��B	�C(����33�c�
@Z=qB
��C>aH                                    BxwtX�  �          @�Q���ff?E�@j=qB  C*����ff��@c33B�CA��                                    Bxwtg�  �          @�����ff?@  @j�HBp�C*����ff����@b�\B�CBJ=                                    Bxwtv:  �          @�\)���?+�@p  B=qC+�R������@e�B�CC��                                    Bxwt��  �          @����?��@l(�B
=C,�R�����@_\)Bp�CC��                                    Bxwt��  �          @�=q���?z�@{�B"33C,�=��녿�p�@l(�B=qCF�                                    Bxwt�,  �          @�����?
=q@l��Bz�C-�����Ϳ��@^{B	  CC޸                                    Bxwt��  �          @�p���=q?��@r�\B{C(s3��=q���\@s33BG�C?aH                                    Bxwt�x  �          @ƸR����?���@]p�BffC&�\���׿5@eBQ�C<�                                    Bxwt�  �          @�\)��=q?��@q�B�C'�)��=q��  @s33B�RC?�                                    Bxwt��  �          @�G���Q�?�z�@aG�B=qC$E��Q�\)@p  B�\C:^�                                    Bxwt�j  �          @����\?(��@a�B(�C,�3���\���H@XQ�A���C@��                                    Bxwt�  �          @�  ��z�?(�@dz�BG�C-����zΰ�
@X��A�G�CAW
                                    Bxwu�  �          @��
��  ?�
=@Y��B{C'T{��  �333@a�B��C;��                                    Bxwu\  �          @�G���{?�
=@UB �C':���{�.{@^�RB�C;u�                                    Bxwu&  �          @�G����?��@\(�B�C(�)����W
=@_\)B��C=O\                                    Bxwu4�  �          @ə���(�?��H@Z=qB�\C&Ǯ��(��0��@c33B	C;�H                                    BxwuCN  �          @Ǯ���\?}p�@tz�B  C(h����\��=q@r�\B�HC@�)                                    BxwuQ�  �          @��H��\)?��@hQ�BC%:���\)�5@s33B(�C<)                                    Bxwu`�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxwuo@   1          @�=q���?Q�@�  B(�C)޸�������@��\B �RCEn                                    Bxwu}�  �          @��
���?fff@�G�B)��C(�q�����z�@���B"��CD��                                    Bxwu��  �          @�{��Q�?k�@�
=B$(�C)���Q쿬��@��B�CC�
                                    Bxwu�2  �          @����p�?0��@�G�Bp�C+����p����H@u�B
=CDxR                                    Bxwu��  �          @�����  ?&ff@�Q�B2��C+����  ��p�@��RB$p�CI�                                    Bxwu�~  �          @��
���
>�  @Z�HB
=C1aH���
���
@Dz�A�RCC��                                    Bxwu�$  �          @ʏ\��p�?�@e�B
33C-����p���=q@XQ�B(�CBp�                                    Bxwu��  �          @����?J=q@`��BffC+T{����{@Z�HB��C@{                                    Bxwu�p  �          @ʏ\��
=?   @c33BffC.�{��
=����@S�
A��CB�{                                    Bxwu�  �          @�p���  >��R@e�B�HC0����  �Ǯ@O\)A�\CD�                                    Bxwv�  �          @�p�����?��\@aG�B  C)����׿h��@c33BG�C=�
                                    Bxwvb  �          @�=q����?\(�@s�
B{C*Ǯ���׿��H@n{B	��C@�f                                    Bxwv  �          @�����  >��H@]p�B \)C.�f��  ���@N�RA�CA��                                    Bxwv-�  �          @����(�>�
=@mp�B
  C/���(����
@Z=qA���CC�H                                    Bxwv<T  
�          @�33��33>Ǯ@u�B�C/޸��33��\)@_\)A��
CD޸                                    BxwvJ�  �          @У�����>���@s�
B�HC0h����ÿ�z�@\(�A��HCEp�                                    BxwvY�  �          @ҏ\����>�G�@y��B��C/E���׿�\)@e�B��CE\                                    BxwvhF  �          @��
��G�?5@{�B=qC,Q���G����@o\)B	Q�CB�q                                    Bxwvv�  �          @�=q��?�@i��B�C.n����z�@Z=qA�(�CB�\                                    Bxwv��  �          @��
��G�?=p�@�ffB��C+����G���G�@\)B��CD��                                    Bxwv�8  �          @�=q��Q�?��@�z�B((�C'aH��Q쿪=q@�=qB$�CC��                                    Bxwv��  �          @�33��=q?^�R@��B+{C)G���=q���H@�z�B#(�CE�3                                    Bxwv��  �          @�Q���ff?#�
@��\B/�\C+�\��ff��z�@�G�B!��CH�                                     Bxwv�*  �          @�33���H?��H@�
=B&��C%8R���H��\)@�  B'��CA�q                                    Bxwv��  �          @�(�����?�{@���B'�C#O\������G�@��B,ffC@�\                                    Bxwv�v  �          @�p����?���@�G�B'�
C W
����Q�@�  B2p�C>W
                                    Bxwv�  �          @�����?��R@�G�B'p�C!޸�����k�@�ffB/{C?n                                    Bxwv��  T          @�{��(�?�@�z�B8�C�R��(��p��@��HBC=qC@��                                    Bxww	h  �          @����~{?�{@��B?��C�
�~{��ff@���BH=qCB�                                    Bxww  �          @�  ��{>�{@aG�BQ�C0n��{��G�@L(�A��
CC�=                                    Bxww&�  �          @�{��>�@q�B  C.������ff@^�RB  CD�H                                    Bxww5Z  �          @�����>�@��\B��C.�����ٙ�@o\)B  CG(�                                    BxwwD   �          @˅����?@  @\)B\)C+=q���Ϳ�33@s�
B33CC�R                                    BxwwR�  �          @ȣ����
>�{@I��A���C0h����
��=q@8Q�A��CA�                                    BxwwaL  �          @�p���>��
@S33B��C0z�����@?\)A���CCT{                                    Bxwwo�  �          @�33��Q�?   @AG�A��C.�)��Q쿏\)@5A���C@\                                    Bxww~�  �          @�\)��z�>�33@*�HA�
=C0����zῈ��@{A��C>W
                                    Bxww�>  �          @�z�����>�z�@(�A��C1#����Ϳ�G�@  A���C=��                                    Bxww��  �          @������?
=q@ffA�Q�C.�������R@�A�z�C9�                                    Bxww��  T          @���33?��@%A��C.����33�\(�@   A�\)C<W
                                    Bxww�0  �          @ə���
=?@  @ ��A�ffC,�\��
=�&ff@"�\A�z�C:�                                    Bxww��  �          @�p����\?333@\)A�G�C,�{���\�.{@   A�C:�R                                    Bxww�|  �          @������H?��@p�A��
C(� ���H�.{@�RA��C5�3                                    Bxww�"  �          @�\)���
?��@��A���C&�H���
���
@(Q�A�Q�C433                                    Bxww��  �          @�������?5@�\A�33C-\������@�A�{C9�f                                    Bxwxn  �          @�{������?���AK�C5z�����Tz�?�=qA"{C;��                                    Bxwx  �          @�\)��p����?��A�
=C6���p���Q�?�p�A\(�C?ff                                    Bxwx�  �          @�����H�L��?���A��C4z����H��ff?��Aw�
C>&f                                    Bxwx.`  T          @�=q����>8Q�@�HA��RC2����ÿ��@
�HA�{C?!H                                    Bxwx=  T          @��H����=�Q�@@��A�33C3{���ÿ�p�@'�A�Q�CC�H                                    BxwxK�  �          @�33����<�@!�A��
C3���������
@�A�p�C@�q                                    BxwxZR  T          @�ff��  �
=q@��A��RC9aH��  �У�?�33Aw�CC�\                                    Bxwxh�  �          @��H����J=q?�
=A���C;�H�����ff?��
A��CC�                                    Bxwxw�  �          @�{���R>��?\An�RC2� ���R�(��?���AX(�C:�\                                    Bxwx�D  �          @������O\)?���A>�\C<.������?(�@���CA!H                                    Bxwx��  �          @�G����Ϳz�H?(��@�Q�C=k����Ϳ�
==u?
=qC?G�                                    Bxwx��  �          @Å��{��  �����\)C?�H��{��  �E�����C=��                                    Bxwx�6  �          @�(������E����
�B�\C;@ ������R�����z�C9�{                                    Bxwx��  �          @�(�����aG�?�33A-C6�����G�?^�RA33C;p�                                    Bxwxς  �          @�\)���ÿ
=?�=qA!p�C9�����ÿ��?�R@���C=��                                    Bxwx�(  �          @�33��{����>��
@<(�C>Y���{��녾��R�2�\C>c�                                    Bxwx��  �          @�33��(����;B�\��(�CB����(����\�}p���
C?��                                    Bxwx�t  �          @�ff��  �Ǯ=#�
>�{CB���  ��{�G���p�C@E                                    Bxwy
  �          @У��˅��z�=#�
>�p�C@� �˅��p��0����33C>�                                    Bxwy�  T          @�z����ÿ333?�@�Q�C:
���ÿ\(�=��
?8Q�C;�=                                    Bxwy'f  �          @ҏ\�θR?
=q?��A=qC/@ �θR���
?�A$Q�C4�3                                    Bxwy6  �          @�p��ə�?�?p��A�HC/E�ə��#�
?���AffC4Y�                                    BxwyD�  �          @�\)��(�?z�?uA\)C/���(�    ?�\)A��C4�                                    BxwySX  �          @�  �ָR>\)>��@�G�C2��ָR��>�@�33C5
=                                    Bxwya�  �          @�����  >�(�>k�?�
=C0c���  >�  >�
=@aG�C1��                                    Bxwyp�  �          @����z�>aG��������C2+���z�>�{�\)��33C1(�                                    BxwyJ  �          @��H��G�=�Q�(�����C35���G�>��;���  C0�f                                    Bxwy��  �          @�{��G�>aG�����8��C2
=��G�?aG����
���C,O\                                    Bxwy��  �          @�{���
?:�H�������C-.���
?�ff��33�%C%                                    Bxwy�<  �          @�ff��33?�������RC'h���33@�����A�C&f                                    Bxwy��  �          @�����H?��G���(�C#aH���H@&ff��=q��HC�                                    BxwyȈ  �          @ə���Q�?�z��(���(�C#33��Q�@,(���p��5C�q                                    Bxwy�.  "          @\��  ?�녿��H�9��C&c���  ?��þ��
�AG�C"p�                                    Bxwy��  �          @�����=q?(�ÿ����K�
C-� ��=q?�  �J=q���C'ٚ                                    Bxwy�z  �          @�������?@  �����P  C,�
����?���B�\��=qC&�3                                    Bxwz   �          @�  ���?z�H�����C*n���?�\)�\�h��C&��                                    Bxwz�  �          @\���?�G��У��~{C)����?��H�aG��\)C#5�                                    Bxwz l  �          @�33��G�@<�ͿE��߮C��G�@:�H?^�R@�(�C=q                                    Bxwz/  �          @�
=���?��H�z�H�(�C%�����?�G����
�G�C"�q                                    Bxwz=�  �          @\����@8Q�����u�CW
����@*�H?�33A/�C)                                    BxwzL^  T          @�33���R@&ff�.{��C�=���R@$z�?B�\@�ffC��                                    Bxwz[  �          @θR��?��H�\(���  C&� ��?�Q�<#�
>�C$�                                    Bxwzi�  �          @�33���?���n{�(�C#aH���@ ��>�?��C!k�                                    BxwzxP  �          @�ff���R@:�H��G��>�\C� ���R@I��>�(�@�=qCٚ                                    Bxwz��  T          @������H@&ff�z�H��C����H@/\)?�\@���C�f                                    Bxwz��  �          @޸R��(�@��
>Ǯ@L��C����(�@b�\@%A�p�C��                                    Bxwz�B  �          @��H���H@`  ?�A��C.���H@�\@X��B33C.                                    Bxwz��  �          @׮���@���?\(�@��HC�����@AG�@5�A�
=CaH                                    Bxwz��  �          @�
=����@vff?�Q�A)�C}q����@+�@@��A���C�H                                    Bxwz�4  �          @���33@:�H?�=qAE�Ck���33?�@)��A���C!\)                                    Bxwz��  �          @�(��Å@
=q?�Q�A&{C }q�Å?��R@��A�\)C(��                                    Bxwz�  �          @ʏ\���@{?�ffAc�C33���?��@'
=A�
=C&�{                                    Bxwz�&  �          @�(����@*=q?�G�An{CG����?�\)@8Q�A��C'G�                                    Bxw{
�  �          @�ff�\@.{?�ffAqp�C�H�\?�33@<(�A�(�C'
=                                    Bxw{r  �          @����\@�?�Q�A�Q�C5��\?���@:�HAʣ�C)�q                                    Bxw{(  �          @�\)����@�R@{A�G�C�R����?B�\@C33A�C,�                                    Bxw{6�  �          @��R��
=?�p�@>{A�  C���
=>#�
@dz�B(�C1�R                                    Bxw{Ed  �          @�ff��33@'�@!G�A�\)C����33?p��@aG�BQ�C'�\                                    Bxw{T
  �          @�  ��@#�
@%A��HC�q��?Y��@b�\B�C)&f                                    Bxw{b�  �          @�Q����@!�@1G�A�ffC�
���?:�H@k�B(�C*p�                                    Bxw{qV  �          @�\)���@[�?��HA�z�C�\���@33@N�RBp�C�\                                    Bxw{�  �          @�G��Z�H@���<�>��RC z��Z�H@j=q@  A�\)C�                                    Bxw{��  �          @�ff�H��@g
=@%A�C{�H��?�  @��\BB�C�3                                    Bxw{�H  �          @�\)�hQ�@�z�?�\)A`  CB��hQ�@5�@S�
B  C�                                    Bxw{��  �          @�(��\��@�p�>B�\?��B��H�\��@y��@%�A��HC}q                                    Bxw{��  �          @��
��@�Q�?L��@�=qB����@��H@Y��B33B�                                     Bxw{�:  �          @�Q쿷
=@�\)����z�B�Q쿷
=@�@(�A�\)B�p�                                    Bxw{��  �          @��׿��
@�
=������B�G����
@�p�@�A�{B֔{                                    Bxw{�  �          @����%@���?�(�A?�B�G��%@q�@mp�B{B��q                                    Bxw{�,  �          @���	��@��R?�=qAzffB���	��@h��@���B.��B�.                                    Bxw|�  �          @��H�)��@�ff?�p�A�Q�B�\�)��@c�
@�p�B,��B�u�                                    Bxw|x  T          @����(Q�@���?�ffA���B���(Q�@^{@�ffB/�B�33                                    Bxw|!  �          @���1G�@�(�?�Q�A�  B�33�1G�@S�
@~{B*ffC�q                                    Bxw|/�  �          @����C�
@��@��A�
=B����C�
@33@��B@  C{                                    Bxw|>j  �          @���-p�@�=q?�(�A�  B���-p�@O\)@}p�B,��C޸                                    Bxw|M  �          @�33�Tz�@��?�A�{B���Tz�@2�\@u�B%�C�R                                    Bxw|[�  �          @�33�u@��H?�=qA|Q�C0��u@,(�@^{B
=C
=                                    Bxw|j\  �          @����r�\@z�H@\)A���C{�r�\@��@|(�B'�
C��                                    Bxw|y  �          @����`  @z=q@,��A��C�
�`  ?���@��\B<�HC��                                    Bxw|��  �          @�(��dz�@{�@A�A�\)C@ �dz�?�@��
BDp�C+�                                    Bxw|�N  �          @�p��`��@��\?��
A���B����`��@@��@x��B (�Ck�                                    Bxw|��  �          @\�a�@u�@9��A���C���a�?��
@�ffBA�RC(�                                    Bxw|��  �          @���i��@U@W
=B�
C	}q�i��?���@��BI\)C"�=                                    Bxw|�@  �          @�  �j�H@�{�8Q���B���j�H@��\@��A�p�CE                                    Bxw|��  T          @θR�U�@��׿�z��n�HB����U�@��
?��A8��B��                                    Bxw|ߌ  �          @����7�@������5�B�=q�7�@�ff?�33A���B�=                                    Bxw|�2  �          @أ��3�
@��R���
�Q�B�\�3�
@�z�?�G�AqG�B�
=                                    Bxw|��  �          @Ϯ��@�\)�1G���(�B݊=��@��
>�ff@�  Bم                                    Bxw}~  �          @�33�}p�@�>�\)@��B�Ǯ�}p�@�G�@:=qAӮC��                                    Bxw}$  �          @�z��_\)@�(��333��
=B���_\)@�\)@
�HA�\)B�#�                                    Bxw}(�  �          @����@��@�(���\�w�
B�Q��@��@��?���A@z�B�k�                                    Bxw}7p  �          @θR�J=q@�  �$z���  B����J=q@��>�p�@QG�B���                                    Bxw}F  �          @�z��I��@��H�N{��\)B�\�I��@�녾�=q��B��f                                    Bxw}T�  �          @ҏ\�G�@��H��(��{B����G�@�(���Q��Ip�B��                                    Bxw}cb  �          @���	��@���=q�%  B��	��@��׿��I�B�G�                                    Bxw}r  	�          @�=q��{@�(��w
=�33BѨ���{@�z�.{���B�                                    Bxw}��  T          @��;L��@hQ���z��[  B�.�L��@�Q��#33���RB��f                                    Bxw}�T  �          @�ff�c�
@~{��
=�J��B�G��c�
@��R�{��  B�                                      Bxw}��  �          @�\)=��
@z�H��G��J�
B��3=��
@�=q�ff��ffB�u�                                    Bxw}��  �          @��?��@�ff��G��M33B�\?��@�=q�Q���z�B�
=                                    Bxw}�F  �          @޸R>�z�@�
=���\�GB��>�z�@ҏ\������B�                                    Bxw}��  �          @�Q콣�
@�{��
=�@33B�\���
@�ff�z���G�B�p�                                    Bxw}ؒ  "          @�33���\@�p����R�2(�B�\���\@\�Ǯ�c�B�{                                    Bxw}�8  "          @�ff��z�@�\)����3G�B����z�@ҏ\���
�m��B�#�                                    Bxw}��  
�          @����
@�����7=qB����
@�ff�
�H���B���                                    Bxw~�  
Z          @�
=���@��\����I33Bה{���@�z����G�B�
=                                    Bxw~*  
�          @�녿�  @x������M�\B׀ ��  @��R������
B˙�                                    Bxw~!�  T          @�  ����@qG����
�L�RB�����@���  ���HB�33                                    Bxw~0v  �          @�녿���@o\)��{�Ap�B������@��
����p�B�z�                                    Bxw~?  �          @�Q쿺�H@Y�����\�\�B�\���H@����5��p�B�                                    Bxw~M�  !          @��#�
@��\���R�N\)B��H�#�
@�p������Q�B��H                                    Bxw~\h  �          @���>.{@vff��=q�fffB��f>.{@���P  �֣�B��                                    Bxw~k  "          @�(�?B�\@J�H����� B��?B�\@�  ����	��B�33                                    Bxw~y�  �          A�\?��@<������k�B��f?��@�z����R��B�8R                                    Bxw~�Z  �          @�?�\)?�\)����BlQ�?�\)@�  ���\�3\)B��                                    